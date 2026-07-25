"""Top-11 rolling pairs with equal-dollar sizing on both legs."""

import numpy as np


N_INSTRUMENTS = 51
LOOKBACK_DAYS = 250
ENTRY_Z = 0.50
EXIT_Z = 0.10
DOLLAR_LIMIT = 10_000

PAIRS = (
    (37, 25, 'EELT', 'CTGI'), (45, 13, 'NGTE', 'EORC'),
    (18, 35, 'RTTH', 'NAYO'), (10, 46, 'SMAH', 'ILVX'),
    (40, 7, 'ULXY', 'HETT'), (49, 50, 'MHRM', 'EAFC'),
    (20, 1, 'NWIG', 'AENO'), (8, 27, 'HUXZ', 'ACAC'),
    (33, 12, 'MTNS', 'MSDP'), (31, 43, 'ACIX', 'ITPA'),
    (41, 36, 'BLBT', 'FWWG'),
)

pair_state = np.zeros(len(PAIRS), dtype=int)
last_history_length = None


def reset_state():
    global pair_state, last_history_length
    pair_state = np.zeros(len(PAIRS), dtype=int)
    last_history_length = None


def getMyPosition(prcSoFar):
    global last_history_length
    prices = np.asarray(prcSoFar, dtype=float)
    if last_history_length is not None and prices.shape[1] != last_history_length + 1:
        reset_state()
    last_history_length = prices.shape[1]
    if prices.shape[1] < LOOKBACK_DAYS or np.any(prices <= 0) or not np.all(np.isfinite(prices)):
        return np.zeros(N_INSTRUMENTS, dtype=int)

    window, current = prices[:, -LOOKBACK_DAYS:], prices[:, -1]
    target = np.zeros(N_INSTRUMENTS, dtype=int)
    for i, (a, b, *_) in enumerate(PAIRS):
        log_a, log_b = np.log(window[a]), np.log(window[b])
        beta = np.polyfit(log_b, log_a, 1)[0]
        spread = log_a - beta * log_b
        std = spread.std()
        if std < 1e-12:
            pair_state[i] = 0
            continue
        z_score = (np.log(current[a]) - beta * np.log(current[b]) - spread.mean()) / std
        state = pair_state[i]
        if state == 0:
            state = 1 if z_score < -ENTRY_Z else (-1 if z_score > ENTRY_Z else 0)
        elif state == 1 and z_score > -EXIT_Z:
            state = 0
        elif state == -1 and z_score < EXIT_Z:
            state = 0
        pair_state[i] = state
        if state:
            # Equal-dollar: beta defines the signal only; each leg targets $10k.
            target[a] += int(state * DOLLAR_LIMIT / current[a])
            target[b] += int(-state * DOLLAR_LIMIT / current[b])

    limits = (DOLLAR_LIMIT / current).astype(int)
    target = np.clip(target, -limits, limits).astype(int)
    target[0] = 0
    return target
