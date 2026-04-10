"""#10: ML-based exit timing model.

Trains a gradient boosted tree on historical trade checkpoints to predict:
"Given the current state of this position, should I exit now?"

Features at each daily checkpoint:
- Day of week (0-4)
- Unrealized P&L as % of max profit
- Unrealized P&L as % of max loss
- Current delta of the position
- Current gamma
- Underlying momentum (1d, 3d)
- VIX level
- VIX change today
- Days remaining to expiry

Target: whether exiting NOW produces better outcome than holding to next checkpoint.

Uses scikit-learn if available, falls back to a rule-based heuristic.
"""

from dataclasses import dataclass, field
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

MODEL_PATH = Path.home() / ".weeklyopt_cache" / "ml_exit_model.pkl"


@dataclass
class ExitCheckpoint:
    """Feature snapshot at a daily position checkpoint."""
    day_of_week: int  # 0=Mon, 4=Fri
    days_remaining: int
    unrealized_pnl_pct: float  # vs max profit
    unrealized_loss_pct: float  # vs max loss
    position_delta: float
    position_gamma: float
    underlying_1d_return: float
    underlying_3d_return: float
    vix_level: float
    vix_1d_change: float
    iv_rank: float
    # Label
    should_exit: bool = False  # True if exiting now is better than holding


@dataclass
class MLExitModel:
    """Train and use ML model for exit timing."""
    model: object = None
    is_trained: bool = False
    feature_names: list = field(default_factory=lambda: [
        "day_of_week", "days_remaining", "unrealized_pnl_pct",
        "unrealized_loss_pct", "position_delta", "position_gamma",
        "underlying_1d_return", "underlying_3d_return",
        "vix_level", "vix_1d_change", "iv_rank",
    ])

    def train(self, checkpoints: list[ExitCheckpoint], verbose: bool = True) -> float:
        """Train the exit model on historical checkpoint data.

        Returns cross-validated accuracy.
        """
        if not HAS_SKLEARN:
            if verbose:
                print("  scikit-learn not installed. Using rule-based fallback.")
                print("  Install with: pip install scikit-learn")
            return 0.0

        if len(checkpoints) < 50:
            if verbose:
                print(f"  Not enough data to train ({len(checkpoints)} checkpoints, need 50+)")
            return 0.0

        X = np.array([[
            cp.day_of_week, cp.days_remaining, cp.unrealized_pnl_pct,
            cp.unrealized_loss_pct, cp.position_delta, cp.position_gamma,
            cp.underlying_1d_return, cp.underlying_3d_return,
            cp.vix_level, cp.vix_1d_change, cp.iv_rank,
        ] for cp in checkpoints])

        y = np.array([cp.should_exit for cp in checkpoints]).astype(int)

        if verbose:
            print(f"  Training on {len(checkpoints)} checkpoints...")
            print(f"  Exit rate in data: {y.mean():.1%}")

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )

        # Cross-validate
        scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
        accuracy = scores.mean()

        if verbose:
            print(f"  CV Accuracy: {accuracy:.1%} (+/- {scores.std():.1%})")

        # Train on full data
        self.model.fit(X, y)
        self.is_trained = True

        if verbose:
            # Feature importance
            importances = list(zip(self.feature_names, self.model.feature_importances_))
            importances.sort(key=lambda x: -x[1])
            print(f"  Top features:")
            for name, imp in importances[:5]:
                print(f"    {name:>25s}: {imp:.3f}")

        return accuracy

    def predict_exit(self, checkpoint: ExitCheckpoint) -> tuple[bool, float]:
        """Predict whether to exit now.

        Returns (should_exit, confidence).
        """
        if not self.is_trained or self.model is None:
            return self._rule_based_exit(checkpoint)

        X = np.array([[
            checkpoint.day_of_week, checkpoint.days_remaining,
            checkpoint.unrealized_pnl_pct, checkpoint.unrealized_loss_pct,
            checkpoint.position_delta, checkpoint.position_gamma,
            checkpoint.underlying_1d_return, checkpoint.underlying_3d_return,
            checkpoint.vix_level, checkpoint.vix_1d_change, checkpoint.iv_rank,
        ]])

        proba = self.model.predict_proba(X)[0]
        should_exit = proba[1] > 0.55  # slight bias toward holding (avoid over-trading)
        confidence = max(proba)

        return should_exit, confidence

    def _rule_based_exit(self, cp: ExitCheckpoint) -> tuple[bool, float]:
        """Fallback: heuristic exit rules when ML model isn't available."""
        # Thursday or later: exit
        if cp.days_remaining <= 1:
            return True, 0.9

        # Profit target: >50% of max profit captured
        if cp.unrealized_pnl_pct > 0.50:
            return True, 0.8

        # Stop loss: losing >150% of credit received
        if cp.unrealized_loss_pct > 1.5:
            return True, 0.85

        # High gamma + near expiry = dangerous
        if abs(cp.position_gamma) > 0.08 and cp.days_remaining <= 2:
            return True, 0.7

        # VIX spiking + position losing = get out
        if cp.vix_1d_change > 0.15 and cp.unrealized_pnl_pct < -0.20:
            return True, 0.75

        return False, 0.6

    def save(self) -> Path:
        """Save trained model to disk."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "is_trained": self.is_trained,
                "feature_names": self.feature_names,
            }, f)
        return MODEL_PATH

    @classmethod
    def load(cls) -> "MLExitModel":
        """Load trained model from disk."""
        ml = cls()
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                ml.model = data["model"]
                ml.is_trained = data["is_trained"]
                ml.feature_names = data.get("feature_names", ml.feature_names)
            except Exception:
                pass
        return ml


def generate_training_data(
    trades_df: pd.DataFrame,
    equity_data: dict[str, pd.DataFrame],
    vix_data: pd.DataFrame | None = None,
) -> list[ExitCheckpoint]:
    """Generate training checkpoints from historical backtest trades.

    For each trade, simulate daily checkpoints and label:
    should_exit = True if exiting now gives better P&L than holding to next day.
    """
    checkpoints = []

    for _, trade in trades_df.iterrows():
        ticker = trade["ticker"]
        if ticker not in equity_data:
            continue

        df = equity_data[ticker]
        entry_dt = trade["entry_date"]
        exit_dt = trade["exit_date"]

        week_dates = df.index[(df.index >= entry_dt) & (df.index <= exit_dt)]
        if len(week_dates) < 3:
            continue

        final_pnl = trade["pnl"]
        max_risk = trade["max_risk"] if trade["max_risk"] > 0 else 1

        for day_idx in range(1, len(week_dates) - 1):
            dt = week_dates[day_idx]
            days_left = len(week_dates) - day_idx - 1

            # Approximate unrealized P&L at this point (linear interpolation)
            progress = day_idx / (len(week_dates) - 1)
            # Rough: P&L accrues non-linearly but we approximate
            current_pnl_est = final_pnl * progress * (1 + 0.3 * (1 - progress))  # front-loaded for theta

            unrealized_pnl_pct = current_pnl_est / max_risk if max_risk > 0 else 0
            unrealized_loss_pct = -current_pnl_est / max_risk if current_pnl_est < 0 else 0

            # Underlying returns
            if day_idx >= 1:
                ret_1d = float(df.loc[dt, "Close"] / df.loc[week_dates[day_idx-1], "Close"] - 1)
            else:
                ret_1d = 0
            if day_idx >= 3:
                ret_3d = float(df.loc[dt, "Close"] / df.loc[week_dates[day_idx-3], "Close"] - 1)
            else:
                ret_3d = 0

            # VIX
            vix_level = 20.0
            vix_change = 0.0
            if vix_data is not None and not vix_data.empty:
                vix_hist = vix_data.loc[:dt, "Close"]
                if len(vix_hist) >= 2:
                    vix_level = float(vix_hist.iloc[-1])
                    vix_change = float(vix_hist.iloc[-1] / vix_hist.iloc[-2] - 1)

            # Label: should we have exited here?
            # Compare P&L if we exit now vs hold to end
            remaining_pnl = final_pnl - current_pnl_est
            should_exit = remaining_pnl < 0  # if holding makes it worse, we should have exited

            cp = ExitCheckpoint(
                day_of_week=dt.weekday(),
                days_remaining=days_left,
                unrealized_pnl_pct=unrealized_pnl_pct,
                unrealized_loss_pct=unrealized_loss_pct,
                position_delta=0.0,  # would need live Greeks computation
                position_gamma=0.0,
                underlying_1d_return=ret_1d,
                underlying_3d_return=ret_3d,
                vix_level=vix_level,
                vix_1d_change=vix_change,
                iv_rank=50.0,  # would need per-ticker IV rank at that date
                should_exit=should_exit,
            )
            checkpoints.append(cp)

    return checkpoints
