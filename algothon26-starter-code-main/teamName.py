"""Plateau-tuned 12-pair + |y|-weighted Ridge, PLUS an RRES-spread fill.

Identical to the submission (teamName.py) EXCEPT for one addition:

  * When the RTTH-NAYO pair (18,35) is FLAT, its two stocks would normally be
    handed to the Ridge (free_pair_stocks). But RRES (24) LEADS RTTH (+0.118)
    and NOT NAYO (~0), so the RRES signal survives in the RTTH-NAYO spread
    (+0.096) instead of cancelling. We use it: when the pair is flat, trade the
    spread DOLLAR-NEUTRAL (still hedged) in the direction RRES predicts, instead
    of the Ridge fill. The Ridge is a poor predictor of RTTH (IC 0.009), so this
    both removes a weak fill AND adds a real, hedged signal.

    Direction = sign( slope(RRES_t -> spread_return_{t+1}) * RRES_last ), fit on a
    trailing 250-day window. Only the RTTH & NAYO legs are affected, and only
    when that one pair is flat (~20% of days).

Validated: +15 on 500-750 (829 -> 844), and wins 11/11 rolling 250-day windows
across 1-750 (avg +19).

Selected parameters (plateau-tuned on the full 1-750 sample):
    pairs: 241-day lookback, 0.5/0.25 entry/exit z, 50-day time stop
    Ridge: fixed alpha 800, |y|-weighted, top/bottom k targets, $10,000 each
"""

import numpy as np


N_INSTRUMENTS = 51
ALGO_INDEX = 0
NON_ALGO_INDICES = np.arange(1, N_INSTRUMENTS, dtype=int)
DLR_LIMIT = 10_000
EPS = 1e-12

PAIR_LOOKBACK_DAYS = 241
PAIR_ENTRY_Z = 0.5
PAIR_EXIT_Z = 0.25
PAIR_MAX_HOLDING_DAYS = 50   # plateau-retuned on full 1-750 (was 30)

RIDGE_MIN_TRAINING_OBSERVATIONS = 120
RIDGE_ALPHA = 800.0          # plateau-retuned on full 1-750 (was 702)
RIDGE_TAIL_SIZE = 15
RIDGE_DOLLARS_PER_POSITION = 10_000

# Spread is log(A) - beta * log(B).
PAIRS = [
    (31, 43),  # ACIX-ITPA
    (49, 50),  # MHRM-EAFC
    (25, 37),  # CTGI-EELT
    (1, 20),   # AENO-NWIG
    (36, 41),  # FWWG-BLBT
    (10, 46),  # SMAH-ILVX
    (13, 45),  # EORC-NGTE
    (8, 27),   # HUXZ-ACAC
    (40, 7),   # ULXY-HETT
    (18, 35),  # RTTH-NAYO
    (33, 42),  # MTNS-BENI
    (26, 32),  # ALUT-CCNS
]

PAIR_INDICES = {index for pair in PAIRS for index in pair}
RIDGE_TARGET_INDICES = np.array(
    [index for index in NON_ALGO_INDICES if index not in PAIR_INDICES],
    dtype=int,
)

if len(PAIR_INDICES) != 2 * len(PAIRS):
    raise ValueError("Pairs must not share a stock")

# --- RRES-spread fill on the flat RTTH-NAYO pair ---
RTTH_INDEX = 18
NAYO_INDEX = 35
RRES_INDEX = 24
RTTH_NAYO_PAIR_NUMBER = PAIRS.index((RTTH_INDEX, NAYO_INDEX))  # 9
RRES_FIT_WINDOW = 250


currentPos = np.zeros(N_INSTRUMENTS, dtype=int)
pairStates = np.zeros(len(PAIRS), dtype=int)
pairHoldingDays = np.zeros(len(PAIRS), dtype=int)
lastHistoryLength = None


def reset_state():
    """Clear state before starting an independent backtest."""
    global currentPos, pairStates, pairHoldingDays, lastHistoryLength

    currentPos = np.zeros(N_INSTRUMENTS, dtype=int)
    pairStates = np.zeros(len(PAIRS), dtype=int)
    pairHoldingDays = np.zeros(len(PAIRS), dtype=int)
    lastHistoryLength = None


def _fit_pair_parameters(window_prices, idx_a, idx_b):
    log_a = np.log(window_prices[idx_a])
    log_b = np.log(window_prices[idx_b])

    beta = np.polyfit(log_b, log_a, 1)[0]
    spread = log_a - beta * log_b
    return beta, spread.mean(), spread.std()


def _target_pair_position(state, beta, price_a, price_b):
    if state == 0:
        return np.array([0, 0], dtype=int)

    shares_a = int(state * DLR_LIMIT / price_a)
    shares_b = int(-state * DLR_LIMIT / price_b)
    return np.array([shares_a, shares_b], dtype=int)


def _build_pair_positions(prices):
    """Build positions for the stateful pair trades."""
    global pairStates, pairHoldingDays

    positions = np.zeros(N_INSTRUMENTS, dtype=int)
    if prices.shape[1] < PAIR_LOOKBACK_DAYS:
        return positions

    window_prices = prices[:, -PAIR_LOOKBACK_DAYS:]
    current_prices = prices[:, -1]

    for pair_number, (idx_a, idx_b) in enumerate(PAIRS):
        beta, spread_mean, spread_std = _fit_pair_parameters(
            window_prices,
            idx_a,
            idx_b,
        )

        if spread_std < EPS:
            pairStates[pair_number] = 0
            pairHoldingDays[pair_number] = 0
            continue

        spread_now = (
            np.log(current_prices[idx_a])
            - beta * np.log(current_prices[idx_b])
        )
        z_now = (spread_now - spread_mean) / spread_std

        state = pairStates[pair_number]
        next_state = state

        if state == 0:
            if z_now < -PAIR_ENTRY_Z:
                next_state = 1
            elif z_now > PAIR_ENTRY_Z:
                next_state = -1
        elif state == 1 and z_now > -PAIR_EXIT_Z:
            next_state = 0
        elif state == -1 and z_now < PAIR_EXIT_Z:
            next_state = 0

        if (
            state != 0
            and pairHoldingDays[pair_number] >= PAIR_MAX_HOLDING_DAYS
        ):
            next_state = 0

        if next_state == 0:
            pairHoldingDays[pair_number] = 0
        elif state == 0:
            pairHoldingDays[pair_number] = 1
        else:
            pairHoldingDays[pair_number] += 1

        pairStates[pair_number] = next_state
        pair_position = _target_pair_position(
            next_state,
            beta,
            current_prices[idx_a],
            current_prices[idx_b],
        )
        positions[idx_a] += pair_position[0]
        positions[idx_b] += pair_position[1]

    return positions


def _ridge_predict(X_train, y_train, X_predict):
    """|y|-weighted standardized Ridge."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_predict = np.atleast_2d(np.asarray(X_predict, dtype=float))

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0, ddof=0)
    x_std = np.where(x_std < EPS, 1.0, x_std)
    y_mean = y_train.mean()

    X_standardized = (X_train - x_mean) / x_std
    X_predict_standardized = (X_predict - x_mean) / x_std

    weights = np.abs(y_train)
    weights = weights / (weights.mean() + EPS)
    Xw = X_standardized * weights[:, None]

    gram = Xw.T @ X_standardized
    beta = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(gram.shape[0]),
        Xw.T @ (y_train - y_mean),
    )
    return y_mean + X_predict_standardized @ beta


def _forecast_ridge_target(all_returns, target_index):
    predictor_indices = NON_ALGO_INDICES[
        NON_ALGO_INDICES != target_index
    ]

    X_history = all_returns[predictor_indices, :-1].T
    y_history = all_returns[target_index, 1:]
    X_current = all_returns[predictor_indices, -1]

    if len(y_history) < RIDGE_MIN_TRAINING_OBSERVATIONS:
        return None
    if not (
        np.all(np.isfinite(X_history))
        and np.all(np.isfinite(y_history))
        and np.all(np.isfinite(X_current))
    ):
        return None

    return float(_ridge_predict(X_history, y_history, X_current)[0])


def _build_ridge_positions(prices, free_pair_stocks=()):
    """Trade the k highest and k lowest one-day Ridge forecasts, and fill any
    currently-flat pair stocks with a ridge-directed $10k position."""
    positions = np.zeros(N_INSTRUMENTS, dtype=int)
    if prices.shape[1] < RIDGE_MIN_TRAINING_OBSERVATIONS + 2:
        return positions

    all_returns = prices[:, 1:] / prices[:, :-1] - 1.0
    current_prices = prices[:, -1]
    dollars = min(RIDGE_DOLLARS_PER_POSITION, DLR_LIMIT)

    forecasts = {}
    for target_index in RIDGE_TARGET_INDICES:
        forecast = _forecast_ridge_target(all_returns, target_index)
        if forecast is not None and np.isfinite(forecast):
            forecasts[target_index] = forecast

    tail_size = min(RIDGE_TAIL_SIZE, len(forecasts) // 2)
    if tail_size >= 1:
        ranked_targets = sorted(forecasts, key=forecasts.get)
        for target_index in ranked_targets[-tail_size:]:
            positions[target_index] = int(dollars / current_prices[target_index])
        for target_index in ranked_targets[:tail_size]:
            positions[target_index] = -int(dollars / current_prices[target_index])

    for stock in free_pair_stocks:
        forecast = _forecast_ridge_target(all_returns, stock)
        if forecast is not None and np.isfinite(forecast) and forecast != 0.0:
            positions[stock] = int(np.sign(forecast) * dollars / current_prices[stock])

    return positions


def _rres_spread_direction(prices, beta):
    """Sign of the RRES-predicted next move of the RTTH-NAYO spread.

    Fits slope of (RRES return_t -> spread return_{t+1}) on a trailing window,
    then projects it onto the latest RRES return. RRES leads RTTH but not NAYO,
    so this signal survives dollar-neutral in the spread. Returns -1, 0, or +1.
    """
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    spread_returns = returns[RTTH_INDEX] - beta * returns[NAYO_INDEX]

    y = spread_returns[1:]
    x = returns[RRES_INDEX][:-1]
    n = min(len(x), RRES_FIT_WINDOW)
    if n <= 50:
        return 0.0
    x, y = x[-n:], y[-n:]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(np.sign(slope * returns[RRES_INDEX][-1]))


def getMyPosition(prcSoFar):
    """Combined pair + |y|-weighted Ridge positions, with the RRES-spread fill
    on the flat RTTH-NAYO pair."""
    global currentPos, lastHistoryLength

    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2:
        raise ValueError("prcSoFar must be a two-dimensional price array")

    n_instruments, n_days = prices.shape
    if n_instruments != N_INSTRUMENTS:
        raise ValueError(
            f"Expected {N_INSTRUMENTS} instruments, received {n_instruments}"
        )

    if lastHistoryLength is not None and n_days != lastHistoryLength + 1:
        reset_state()
    lastHistoryLength = n_days

    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        currentPos = np.zeros(N_INSTRUMENTS, dtype=int)
        return currentPos

    pair_positions = _build_pair_positions(prices)
    open_stocks = {
        stock
        for pair_number, pair in enumerate(PAIRS)
        if pairStates[pair_number] != 0
        for stock in pair
    }
    free_pair_stocks = [s for s in sorted(PAIR_INDICES) if s not in open_stocks]

    combined_positions = pair_positions + _build_ridge_positions(prices, free_pair_stocks)

    current_prices = prices[:, -1]

    # RRES-spread fill: when the RTTH-NAYO pair is FLAT, replace the ridge fill of
    # its two stocks with a dollar-neutral spread trade in RRES's predicted direction.
    if (
        pairStates[RTTH_NAYO_PAIR_NUMBER] == 0
        and prices.shape[1] >= PAIR_LOOKBACK_DAYS
    ):
        window_prices = prices[:, -PAIR_LOOKBACK_DAYS:]
        beta = np.polyfit(
            np.log(window_prices[NAYO_INDEX]),
            np.log(window_prices[RTTH_INDEX]),
            1,
        )[0]
        direction = _rres_spread_direction(prices, beta)
        if direction != 0.0:
            combined_positions[RTTH_INDEX] = int(direction * DLR_LIMIT / current_prices[RTTH_INDEX])
            combined_positions[NAYO_INDEX] = int(-direction * DLR_LIMIT / current_prices[NAYO_INDEX])

    position_limits = (DLR_LIMIT / current_prices).astype(int)
    combined_positions = np.clip(
        combined_positions,
        -position_limits,
        position_limits,
    ).astype(int)
    combined_positions[ALGO_INDEX] = 0

    currentPos = combined_positions
    return currentPos
