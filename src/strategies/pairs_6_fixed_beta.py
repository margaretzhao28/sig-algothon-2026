"""Simple fixed-beta pairs strategy using six train-selected candidates.

The strategy calibrates each pair once from the history visible at the first
eligible evaluation call, then keeps that hedge beta, spread mean, and spread
standard deviation fixed for the rest of that run.  It is intentionally a
baseline: no rolling refits, no portfolio optimiser, and no ALGO position.
"""

import numpy as np


N_INSTRUMENTS = 51
ALGO_INDEX = 0
LOOKBACK_DAYS = 250
ENTRY_Z = 1.5
EXIT_Z = 0.5
MAX_HOLDING_DAYS = 30
DOLLAR_LIMIT = 10_000
EPS = 1e-12

# Selected from the first-500-day correlation/cointegration research screen.
# Each tuple is (instrument index A, instrument index B, ticker A, ticker B).
PAIRS = (
    (10, 46, "SMAH", "ILVX"),
    (1, 20, "AENO", "NWIG"),
    (13, 45, "EORC", "NGTE"),
    (8, 27, "HUXZ", "ACAC"),
    (49, 50, "MHRM", "EAFC"),
    (25, 37, "CTGI", "EELT"),
)

pair_state = np.zeros(len(PAIRS), dtype=int)
holding_days = np.zeros(len(PAIRS), dtype=int)
pair_parameters = None
last_history_length = None


def reset_state():
    """Reset state before an independent backtest or evaluation run."""
    global pair_state, holding_days, pair_parameters, last_history_length

    pair_state = np.zeros(len(PAIRS), dtype=int)
    holding_days = np.zeros(len(PAIRS), dtype=int)
    pair_parameters = None
    last_history_length = None


def _calibrate(prices):
    """Fit fixed log-price spread parameters on the most recent lookback."""
    parameters = []
    calibration_prices = prices[:, -LOOKBACK_DAYS:]

    for idx_a, idx_b, _, _ in PAIRS:
        log_a = np.log(calibration_prices[idx_a])
        log_b = np.log(calibration_prices[idx_b])
        beta, intercept = np.polyfit(log_b, log_a, 1)
        spread = log_a - (intercept + beta * log_b)
        parameters.append((float(beta), float(intercept), float(spread.mean()), float(spread.std())))

    return parameters


def _target_position(direction, beta, price_a, price_b):
    """Return an integer hedge position while respecting each-leg dollar limits."""
    if direction == 0 or abs(beta) < EPS:
        return np.array([0, 0], dtype=int)

    # Long spread: long one unit of A and short beta units of B.
    unit_a, unit_b = 1.0, -beta
    scale_a = DOLLAR_LIMIT / (abs(unit_a) * price_a)
    scale_b = DOLLAR_LIMIT / (abs(unit_b) * price_b)
    scale = min(scale_a, scale_b)

    return np.array(
        [int(np.round(direction * scale * unit_a)), int(np.round(direction * scale * unit_b))],
        dtype=int,
    )


def getMyPosition(prcSoFar):
    """Return positions for the six fixed-beta mean-reversion pairs."""
    global pair_parameters, last_history_length

    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2 or prices.shape[0] != N_INSTRUMENTS:
        raise ValueError(f"Expected a ({N_INSTRUMENTS}, n_days) price array")

    if last_history_length is not None and prices.shape[1] != last_history_length + 1:
        reset_state()
    last_history_length = prices.shape[1]

    if prices.shape[1] < LOOKBACK_DAYS or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return np.zeros(N_INSTRUMENTS, dtype=int)

    if pair_parameters is None:
        pair_parameters = _calibrate(prices)

    current_prices = prices[:, -1]
    positions = np.zeros(N_INSTRUMENTS, dtype=int)

    for pair_number, (idx_a, idx_b, _, _) in enumerate(PAIRS):
        beta, intercept, spread_mean, spread_std = pair_parameters[pair_number]
        if spread_std < EPS:
            pair_state[pair_number] = 0
            holding_days[pair_number] = 0
            continue

        spread_now = np.log(current_prices[idx_a]) - (intercept + beta * np.log(current_prices[idx_b]))
        z_score = (spread_now - spread_mean) / spread_std
        state = pair_state[pair_number]
        next_state = state

        if state == 0:
            if z_score <= -ENTRY_Z:
                next_state = 1
            elif z_score >= ENTRY_Z:
                next_state = -1
        elif state == 1 and z_score >= -EXIT_Z:
            next_state = 0
        elif state == -1 and z_score <= EXIT_Z:
            next_state = 0

        if state != 0 and holding_days[pair_number] >= MAX_HOLDING_DAYS:
            next_state = 0

        if next_state == 0:
            holding_days[pair_number] = 0
        elif state == 0:
            holding_days[pair_number] = 1
        else:
            holding_days[pair_number] += 1

        pair_state[pair_number] = next_state
        pair_position = _target_position(next_state, beta, current_prices[idx_a], current_prices[idx_b])
        positions[idx_a] += pair_position[0]
        positions[idx_b] += pair_position[1]

    position_limits = (DOLLAR_LIMIT / current_prices).astype(int)
    positions = np.clip(positions, -position_limits, position_limits).astype(int)
    positions[ALGO_INDEX] = 0
    return positions
