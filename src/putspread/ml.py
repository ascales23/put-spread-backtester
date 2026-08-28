"""XGBoost trade filter, trained and evaluated strictly walk-forward.

The problem this addresses: the parameter sweep found no set that generalizes. Rather
than keep searching for a better fixed rule, this learns which *individual* candidate
trades are worth taking, from features available at entry.

Leak control is the whole game here, because the sample is small enough that one
leaked feature would manufacture a spectacular fake edge. Two rules enforce it:

  1. A model that scores a trade entered on day t is trained ONLY on trades that had
     already CLOSED before t. Training on trades that merely *opened* before t leaks
     the future, because their labels were not known yet on day t. With holds of up
     to 60 days this distinction is not academic.
  2. Predictions used by the backtest are out-of-fold by construction. Every trade is
     scored by a model that never saw it, and never saw any trade resolved after it.

Trades in the initial burn-in window have no eligible training data and are simply
not taken. That costs coverage, and pretending otherwise would cost honesty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS

#: Deliberately small-data settings: shallow trees, heavy regularization, strong
#: subsampling. With a few hundred trades, an unconstrained booster memorizes the
#: sample and reports a fictional edge.
DEFAULT_PARAMS = dict(
    max_depth=3,
    n_estimators=250,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=8,
    reg_lambda=3.0,
    reg_alpha=0.5,
    random_state=0,
    n_jobs=4,
)


@dataclass
class FoldSpec:
    """One walk-forward step: train on what had closed, test on what came next."""

    index: int
    train_end: date          # only trades CLOSED strictly before this date may train
    test_start: date
    test_end: date
    n_train: int = 0
    n_test: int = 0


def make_purged_folds(
    df: pd.DataFrame, n_folds: int = 5, min_train: int = 60
) -> list[FoldSpec]:
    """Split by entry date into `n_folds` consecutive test windows.

    A fold is emitted only when at least `min_train` trades had already resolved
    before its test window opens; earlier windows are burn-in and yield no trades.
    """
    if df.empty:
        return []
    entries = pd.to_datetime(df["entry_date"])
    lo, hi = entries.min(), entries.max()
    edges = pd.date_range(lo, hi, periods=n_folds + 1)
    folds = []
    for i in range(n_folds):
        ts, te = edges[i], edges[i + 1]
        train_mask = pd.to_datetime(df["exit_date"]) < ts
        if int(train_mask.sum()) < min_train:
            continue
        test_mask = (entries >= ts) & (entries < te if i < n_folds - 1 else entries <= te)
        if int(test_mask.sum()) == 0:
            continue
        folds.append(FoldSpec(
            index=len(folds) + 1, train_end=ts.date(),
            test_start=ts.date(), test_end=te.date(),
            n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        ))
    return folds


def _matrix(df: pd.DataFrame) -> np.ndarray:
    return df.reindex(columns=FEATURE_COLUMNS).to_numpy(dtype=float)


def fit_model(df_train: pd.DataFrame, target: str, params: dict | None = None):
    """Fit one booster. `target` is 'ror' (regression) or 'win'/'big_loss' (binary)."""
    from xgboost import XGBClassifier, XGBRegressor

    p = {**DEFAULT_PARAMS, **(params or {})}
    X, y = _matrix(df_train), df_train[target].to_numpy(dtype=float)
    if target == "ror":
        model = XGBRegressor(objective="reg:squarederror", **p)
    else:
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **p)
    model.fit(X, y)
    return model


@dataclass
class WalkForwardResult:
    """Out-of-fold predictions plus the per-fold audit trail."""

    predictions: pd.Series               # indexed like the input frame; NaN in burn-in
    folds: list[FoldSpec]
    fold_table: pd.DataFrame
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-row decision threshold, taken from that row's TRAINING fold only. For the
    #: `big_loss` target this is the historical base rate of a tail loss, so the rule
    #: reads "decline any trade whose modelled tail risk is worse than the tail risk
    #: the strategy has historically run". Fixed a priori -- never read off the test
    #: set, which is what makes the deployed number an honest out-of-sample figure.
    thresholds: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def walk_forward_predict(
    df: pd.DataFrame, target: str = "ror", n_folds: int = 5,
    min_train: int = 60, params: dict | None = None,
) -> WalkForwardResult:
    """Score every trade with a model that never saw it, nor anything after it."""
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    folds = make_purged_folds(df, n_folds, min_train)
    entries = pd.to_datetime(df["entry_date"])
    exits = pd.to_datetime(df["exit_date"])
    thresholds = pd.Series(np.nan, index=df.index, dtype=float)
    rows, imps = [], []

    for f in folds:
        train = df[exits < pd.Timestamp(f.train_end)]
        test_mask = (entries >= pd.Timestamp(f.test_start)) & (
            entries <= pd.Timestamp(f.test_end) if f is folds[-1]
            else entries < pd.Timestamp(f.test_end)
        )
        test = df[test_mask]
        if test.empty or train[target].nunique() < 2:
            continue
        model = fit_model(train, target, params)
        pred = (
            model.predict(_matrix(test)) if target == "ror"
            else model.predict_proba(_matrix(test))[:, 1]
        )
        preds.loc[test.index] = pred
        thresholds.loc[test.index] = float(train[target].mean())
        imps.append(pd.Series(model.feature_importances_, index=FEATURE_COLUMNS))
        rows.append({
            "fold": f.index, "train_end": f.train_end, "test_start": f.test_start,
            "test_end": f.test_end, "n_train": len(train), "n_test": len(test),
            "test_mean_ror": float(test["ror"].mean()),
            "test_win_rate": float(test["win"].mean()),
        })

    importance = (
        pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).to_frame("importance")
        if imps else pd.DataFrame()
    )
    return WalkForwardResult(preds, folds, pd.DataFrame(rows), importance, thresholds)


def evaluate_threshold(
    df: pd.DataFrame, preds: pd.Series, threshold: float
) -> dict[str, float]:
    """What taking only `preds >= threshold` would have done, out-of-fold."""
    scored = preds.notna()
    take = scored & (preds >= threshold)
    sel, base = df[take], df[scored]
    if sel.empty:
        return {"threshold": threshold, "trades": 0, "coverage": 0.0}
    gross_win = float(sel.loc[sel["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-sel.loc[sel["pnl"] <= 0, "pnl"].sum())
    return {
        "threshold": threshold,
        "trades": int(len(sel)),
        "coverage": len(sel) / max(len(base), 1),
        "win_rate": float((sel["pnl"] > 0).mean()),
        "expectancy": float(sel["pnl"].mean()),
        "total_pnl": float(sel["pnl"].sum()),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "mean_ror": float(sel["ror"].mean()),
        "worst": float(sel["pnl"].min()),
        "baseline_expectancy": float(base["pnl"].mean()),
        "lift": float(sel["pnl"].mean() - base["pnl"].mean()),
    }


def threshold_scan(
    df: pd.DataFrame, preds: pd.Series, quantiles: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
) -> pd.DataFrame:
    """Out-of-fold performance across selectivity levels.

    A DIAGNOSTIC, not a way to choose the threshold: picking the best row here and
    quoting it would be in-sample selection wearing an out-of-sample costume. The
    deployed rule fixes its threshold from training data only.
    """
    scored = preds.dropna()
    rows = [evaluate_threshold(df, preds, float(scored.quantile(q))) for q in quantiles]
    out = pd.DataFrame(rows)
    out.insert(0, "quantile", list(quantiles))
    return out


class PrecomputedFilter:
    """A `TradeFilter` backed by out-of-fold predictions keyed on (symbol, entry date).

    The harvest enumerates exactly the candidate set the engine will re-derive -- same
    signals, same filters, same fill model -- so the lookup hits by construction. A
    miss means the candidate had no out-of-fold score (burn-in, or an unresolved
    trade at the end of the data), and the trade is DECLINED rather than waved
    through: a filter that silently passes what it cannot score is not a filter.
    """

    def __init__(self, scores: dict[tuple[str, date], float], threshold: float) -> None:
        self.scores = scores
        self.threshold = threshold
        self.missing = 0
        self.declined = 0

    @classmethod
    def from_frame(cls, df: pd.DataFrame, preds: pd.Series, threshold: float) -> "PrecomputedFilter":
        scores = {
            (r.symbol, r.entry_date): float(p)
            for r, p in zip(df.itertuples(index=False), preds)
            if p == p
        }
        return cls(scores, threshold)

    def __call__(self, symbol: str, d: date, candidate) -> tuple[bool, str]:
        score = self.scores.get((symbol, d))
        if score is None:
            self.missing += 1
            return False, "no out-of-fold score (burn-in window)"
        if score < self.threshold:
            self.declined += 1
            return False, f"model score {score:.4f} below {self.threshold:.4f}"
        return True, f"model score {score:.4f}"


#: Config keys the structure selector is allowed to set. Entry-side only: these
#: change which spread is built and nothing about how it is then managed.
STRUCTURE_KEYS = ("short_strike_method", "buffer_pct", "target_dte", "target_short_delta")


class StructureSelector:
    """Picks the best-predicted entry structure for each opportunity.

    For every (symbol, date) the harvest holds one row per structure. This ranks them
    by out-of-fold predicted return-on-risk and returns the winner's config overrides,
    declining the opportunity when even the best structure is not predicted to pay.

    Because the scores are out-of-fold, the structure chosen for a given day was
    chosen by a model that never saw that day, nor any trade that closed after it.
    """

    def __init__(self, choices: dict[tuple[str, date], dict], threshold: float) -> None:
        self.choices = choices
        self.threshold = threshold
        self.declined = 0
        self.missing = 0

    @classmethod
    def from_frame(
        cls, df: pd.DataFrame, preds: pd.Series, threshold: float = 0.0
    ) -> "StructureSelector":
        scored = df.assign(pred=preds).dropna(subset=["pred"])
        choices: dict[tuple[str, date], dict] = {}
        for (sym, d), grp in scored.groupby(["symbol", "entry_date"], sort=False):
            best = grp.loc[grp["pred"].idxmax()]
            if float(best["pred"]) < threshold:
                continue
            overrides = {
                "short_strike_method": "delta" if best["cfg_is_delta_method"] > 0.5 else "buffer",
                "buffer_pct": float(best["cfg_buffer_pct"]),
                "target_dte": int(best["cfg_target_dte"]),
                "target_short_delta": float(best["cfg_target_delta"]),
            }
            choices[(sym, d)] = {"overrides": overrides, "pred": float(best["pred"])}
        return cls(choices, threshold)

    def __call__(self, symbol: str, d: date) -> dict | None:
        hit = self.choices.get((symbol, d))
        if hit is None:
            # Either burn-in, or every structure scored below the bar. Declining is
            # the only honest answer: a selector that falls back to a default when it
            # cannot score is just the default rule wearing a model's name.
            self.missing += 1
            return None
        return hit["overrides"]


def best_per_opportunity(df: pd.DataFrame, preds: pd.Series, threshold: float = 0.0) -> pd.DataFrame:
    """The trade the selector would actually take at each opportunity, with outcome.

    This is the like-for-like comparison against the fixed-structure baseline: same
    opportunities, one trade each, structure chosen out-of-fold.
    """
    scored = df.assign(pred=preds).dropna(subset=["pred"])
    idx = scored.groupby(["symbol", "entry_date"], sort=False)["pred"].idxmax()
    best = scored.loc[idx]
    return best[best["pred"] >= threshold].sort_values("entry_date").reset_index(drop=True)


def apply_tail_rule(res: WalkForwardResult) -> pd.Series:
    """Boolean accept mask for the `big_loss` target: take when modelled tail risk
    sits below the tail risk the training window actually experienced."""
    return res.predictions < res.thresholds


class TailRiskFilter:
    """`TradeFilter` that declines candidates whose modelled tail risk is elevated.

    Keyed on (symbol, entry date) against out-of-fold scores. A candidate with no
    score -- burn-in, or one that never resolved -- is DECLINED, never waved through.
    """

    def __init__(self, accept: dict[tuple[str, date], bool], scores: dict[tuple[str, date], float]):
        self.accept = accept
        self.scores = scores
        self.missing = 0
        self.declined = 0

    @classmethod
    def from_frame(cls, df: pd.DataFrame, res: WalkForwardResult) -> "TailRiskFilter":
        ok = apply_tail_rule(res)
        accept, scores = {}, {}
        for row, a, p in zip(df.itertuples(index=False), ok, res.predictions):
            if p != p:
                continue
            key = (row.symbol, row.entry_date)
            accept[key] = bool(a)
            scores[key] = float(p)
        return cls(accept, scores)

    def __call__(self, symbol: str, d: date, candidate) -> tuple[bool, str]:
        key = (symbol, d)
        if key not in self.accept:
            self.missing += 1
            return False, "no out-of-fold score (burn-in window)"
        if not self.accept[key]:
            self.declined += 1
            return False, f"modelled tail risk {self.scores[key]:.3f} above the base rate"
        return True, f"tail risk {self.scores[key]:.3f}"


class TailRiskSelector:
    """Chooses, per opportunity, the entry structure with the LOWEST modelled tail
    risk, and declines the opportunity when even the safest structure is elevated."""

    def __init__(self, choices: dict[tuple[str, date], dict]):
        self.choices = choices
        self.missing = 0

    @classmethod
    def from_frame(cls, df: pd.DataFrame, res: WalkForwardResult) -> "TailRiskSelector":
        scored = df.assign(pred=res.predictions, thr=res.thresholds).dropna(subset=["pred"])
        choices: dict[tuple[str, date], dict] = {}
        for (sym, d), grp in scored.groupby(["symbol", "entry_date"], sort=False):
            best = grp.loc[grp["pred"].idxmin()]
            if float(best["pred"]) >= float(best["thr"]):
                continue
            choices[(sym, d)] = {
                "overrides": {
                    "short_strike_method": "delta" if best["cfg_is_delta_method"] > 0.5 else "buffer",
                    "buffer_pct": float(best["cfg_buffer_pct"]),
                    "target_dte": int(best["cfg_target_dte"]),
                    "target_short_delta": float(best["cfg_target_delta"]),
                },
                "pred": float(best["pred"]),
            }
        return cls(choices)

    def __call__(self, symbol: str, d: date) -> dict | None:
        hit = self.choices.get((symbol, d))
        if hit is None:
            self.missing += 1
            return None
        return hit["overrides"]
