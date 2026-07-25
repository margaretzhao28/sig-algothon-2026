"""Twelve-pair rolling-beta strategy selected from notebooks 05 and 06."""

import numpy as np

N_INSTRUMENTS = 51
ROLLING_WINDOW_DAYS, ENTRY_Z, EXIT_Z, MAX_HOLDING_DAYS = 250, 0.50, 0.10, 0
DOLLAR_LIMIT, EPS = 10_000, 1e-12
PAIRS = (
    (49, 50, "MHRM", "EAFC"), (31, 43, "ACIX", "ITPA"),
    (10, 46, "SMAH", "ILVX"), (20, 1, "NWIG", "AENO"),
    (8, 27, "HUXZ", "ACAC"), (45, 13, "NGTE", "EORC"),
    (40, 7, "ULXY", "HETT"), (37, 25, "EELT", "CTGI"),
    (33, 12, "MTNS", "MSDP"), (18, 35, "RTTH", "NAYO"),
    (41, 36, "BLBT", "FWWG"), (42, 11, "BENI", "NPCK"),
)
state = held = last_length = None


def reset_state():
    global state, held, last_length
    state = np.zeros(len(PAIRS), dtype=int)
    held = np.zeros(len(PAIRS), dtype=int)
    last_length = None


def getMyPosition(prcSoFar):
    global last_length
    prices = np.asarray(prcSoFar, dtype=float)
    if last_length is not None and prices.shape[1] != last_length + 1:
        reset_state()
    last_length = prices.shape[1]
    if prices.shape[1] < ROLLING_WINDOW_DAYS or np.any(prices <= 0) or not np.all(np.isfinite(prices)):
        return np.zeros(N_INSTRUMENTS, dtype=int)
    window, current = prices[:, -ROLLING_WINDOW_DAYS:], prices[:, -1]
    target = np.zeros(N_INSTRUMENTS, dtype=int)
    for i, (a, b, *_) in enumerate(PAIRS):
        x, y = np.log(window[a]), np.log(window[b])
        beta, intercept = np.polyfit(y, x, 1)
        spread = x - (intercept + beta * y)
        std = spread.std()
        if std < EPS or abs(beta) < EPS:
            state[i] = held[i] = 0
            continue
        z = (np.log(current[a]) - (intercept + beta * np.log(current[b])) - spread.mean()) / std
        next_state = state[i]
        if state[i] == 0:
            next_state = 1 if z <= -ENTRY_Z else (-1 if z >= ENTRY_Z else 0)
        elif state[i] == 1 and z >= -EXIT_Z:
            next_state = 0
        elif state[i] == -1 and z <= EXIT_Z:
            next_state = 0
        if state[i] and MAX_HOLDING_DAYS > 0 and held[i] >= MAX_HOLDING_DAYS:
            next_state = 0
        held[i] = 0 if not next_state else (1 if not state[i] else held[i] + 1)
        state[i] = next_state
        if next_state:
            scale = min(DOLLAR_LIMIT / current[a], DOLLAR_LIMIT / (abs(beta) * current[b]))
            target[a] += int(round(next_state * scale))
            target[b] += int(round(-next_state * beta * scale))
    limits = (DOLLAR_LIMIT / current).astype(int)
    target = np.clip(target, -limits, limits).astype(int)
    target[0] = 0
    return target


reset_state()
