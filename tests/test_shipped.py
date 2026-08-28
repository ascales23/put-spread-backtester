"""The shipped configuration is pinned, so it cannot drift silently."""

from putspread.config import SHIPPED_UNIVERSE, shipped_config


def test_shipped_config_is_what_the_research_chose():
    cfg = shipped_config()
    assert cfg.short_strike_method == "buffer"
    assert cfg.buffer_pct == 0.05
    assert cfg.target_dte == 14
    assert cfg.profit_target_pct == 0.50
    assert cfg.stop_rule == "credit_multiple"
    assert cfg.stop_credit_multiple == 2.5
    assert cfg.time_stop_dte == 3
    assert cfg.symbols == SHIPPED_UNIVERSE


def test_shipped_config_excludes_the_level_break_stop():
    """The level-break stop turned a 92% win rate into 53%. It is the one exit
    conclusion that needs no modelled price to be believed, and it stays off."""
    assert shipped_config().stop_rule != "level_break"


def test_shipped_config_can_be_run_on_real_quotes_only():
    """Every headline number for this config assumes exits may fill at a model mark.
    The conservative bound must stay one flag away, not a code change away."""
    cfg = shipped_config(require_real_quotes_for_exit=True)
    assert cfg.require_real_quotes_for_exit is True
    assert cfg.profit_target_pct == 0.50 and cfg.stop_credit_multiple == 2.5


def test_shipped_leverage_defaults_to_unlevered():
    assert shipped_config().leverage == 1.0
    assert shipped_config(leverage=4.0).leverage == 4.0


def test_shipped_config_overrides_apply():
    cfg = shipped_config(leverage=2.0, require_real_quotes_for_exit=True)
    assert cfg.leverage == 2.0
    assert cfg.require_real_quotes_for_exit is True
    assert cfg.stop_credit_multiple == 2.5     # untouched fields survive
