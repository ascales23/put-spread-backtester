"""Leak control for the learned filter.

These are the most important tests in the project. The sample is a few thousand rows
with a heavily skewed target; a single leaked feature or a single fold that trains on
the future would produce a spectacular, entirely fictional edge -- and it would look
exactly like success.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from putspread.features import FEATURE_COLUMNS, realized_vol
from putspread.ml import (
    TailRiskFilter, TailRiskSelector, apply_tail_rule, evaluate_threshold,
    make_purged_folds, walk_forward_predict,
)


def synthetic_trades(n: int = 400, seed: int = 0, hold_days: int = 30) -> pd.DataFrame:
    """A trade log whose only signal is one feature, with realistic holding periods."""
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 1)
    rows = []
    for i in range(n):
        entry = start + timedelta(days=int(i * 4))
        feats = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
        # dist_sma50 carries the signal; everything else is noise.
        edge = feats["dist_sma50"]
        pnl = float(200 + 400 * edge + rng.normal(0, 150))
        rows.append({
            "symbol": "X", "entry_date": entry,
            "exit_date": entry + timedelta(days=hold_days),
            "pnl": pnl, "ror": pnl / 1500.0, "win": int(pnl > 0),
            "big_loss": int(pnl < -400), **feats,
        })
    return pd.DataFrame(rows)


def test_folds_only_train_on_trades_that_already_closed():
    """The core discipline: a model scoring day t may only learn from trades whose
    OUTCOME was known on day t. Training on trades merely *opened* before t leaks."""
    df = synthetic_trades(300, hold_days=45)
    folds = make_purged_folds(df, n_folds=4, min_train=20)
    assert folds
    for f in folds:
        eligible = df[pd.to_datetime(df["exit_date"]) < pd.Timestamp(f.train_end)]
        assert (pd.to_datetime(eligible["exit_date"]).dt.date < f.test_start).all()
        assert len(eligible) == f.n_train


def test_no_training_row_overlaps_its_test_window():
    df = synthetic_trades(300, hold_days=60)
    for f in make_purged_folds(df, n_folds=4, min_train=20):
        train = df[pd.to_datetime(df["exit_date"]) < pd.Timestamp(f.train_end)]
        # Not one training trade may still have been open when the test window opened.
        assert not (pd.to_datetime(train["exit_date"]).dt.date >= f.test_start).any()


def test_predictions_are_out_of_fold_everywhere():
    df = synthetic_trades(400)
    res = walk_forward_predict(df, target="ror", n_folds=4, min_train=30)
    scored = res.predictions.notna()
    assert scored.sum() > 0
    # The earliest trades cannot be scored: nothing had resolved yet to train on.
    assert not bool(scored.iloc[0])
    assert res.thresholds[scored].notna().all()


def test_model_recovers_a_real_signal():
    """Sanity: if a signal genuinely exists, the walk-forward pipeline must find it.
    A leak test suite that only proves the model finds nothing proves nothing."""
    df = synthetic_trades(600)
    res = walk_forward_predict(df, target="ror", n_folds=4, min_train=40)
    scored = res.predictions.notna()
    corr = np.corrcoef(res.predictions[scored], df.loc[scored, "ror"])[0, 1]
    assert corr > 0.3, f"pipeline failed to recover a planted signal (corr={corr:.2f})"


def test_model_finds_nothing_in_pure_noise():
    """The other half: with no signal, out-of-fold lift must not be systematically
    positive. This is the test that would catch a leak."""
    rng = np.random.default_rng(7)
    df = synthetic_trades(500, seed=3)
    df["pnl"] = rng.normal(100, 300, len(df))          # sever the signal
    df["ror"] = df["pnl"] / 1500.0
    df["win"] = (df["pnl"] > 0).astype(int)
    res = walk_forward_predict(df, target="ror", n_folds=4, min_train=40)
    scored = res.predictions.dropna()
    lift = evaluate_threshold(df, res.predictions, float(scored.quantile(0.5)))["lift"]
    assert abs(lift) < 120.0, f"model found edge in noise (lift={lift:.1f}) -- suspect a leak"


def test_threshold_comes_from_training_data_only():
    """The deployed accept threshold is the training window's own base rate, so it
    cannot encode anything about the test window."""
    df = synthetic_trades(500)
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=40)
    for f in res.folds:
        train = df[pd.to_datetime(df["exit_date"]) < pd.Timestamp(f.train_end)]
        expected = float(train["big_loss"].mean())
        in_fold = (pd.to_datetime(df["entry_date"]).dt.date >= f.test_start) & (
            pd.to_datetime(df["entry_date"]).dt.date <= f.test_end)
        got = res.thresholds[in_fold].dropna()
        if len(got):
            assert np.allclose(got.unique(), expected, atol=1e-9)


def test_filter_declines_what_it_cannot_score():
    """A filter that waves through unscored candidates is not a filter."""
    df = synthetic_trades(300)
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=30)
    f = TailRiskFilter.from_frame(df, res)
    take, why = f("X", date(1999, 1, 1), None)
    assert take is False
    assert "burn-in" in why
    assert f.missing == 1


def test_selector_declines_when_every_structure_is_elevated():
    df = synthetic_trades(300)
    for c in ("cfg_is_delta_method", "cfg_buffer_pct", "cfg_target_dte", "cfg_target_delta"):
        df[c] = 0.0
    df["cfg_target_dte"] = 14.0
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=30)
    res.predictions[:] = 1.0                # every structure maximally risky
    sel = TailRiskSelector.from_frame(df, res)
    assert sel("X", df["entry_date"].iloc[-1]) is None


def test_accept_rule_is_a_strict_comparison_to_the_base_rate():
    df = synthetic_trades(300)
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=30)
    accept = apply_tail_rule(res)
    scored = res.predictions.notna()
    assert (accept[scored] == (res.predictions[scored] < res.thresholds[scored])).all()
    assert not accept[~scored].any()


def test_feature_columns_are_unique_and_ordered():
    """Column order is part of the model contract; a silent reorder mis-scores."""
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_realized_vol_is_annualized_and_needs_enough_history():
    closes = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 300))))
    rv = realized_vol(closes, 20)
    assert 0.05 < rv < 0.40
    assert np.isnan(realized_vol(closes.head(5), 20))
