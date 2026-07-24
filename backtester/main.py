import numpy as np

nInst = 51
currentPos = np.zeros(nInst, dtype=int)

# Track each pair separately so we can apply a time stop pair-by-pair.
pairStates = None
pairHoldingDays = None

# Rolling calibration window
LOOKBACK_DAYS = 200

# Mean-reversion thresholds
ENTRY_Z = 1.0
EXIT_Z = 0.5

# Force-close a trade if it stays open too long
MAX_HOLDING_DAYS = 20

# Official non-ALGO dollar limit
DLR_LIMIT = 10_000

# Candidate pairs from the train-only notebook
# Format: (A_index, B_index)
# Spread is log(A) - beta * log(B)
PAIRS = [
    (33, 42),  # MTNS-BENI
    (31, 43),  # ACIX-ITPA
    (49, 50),  # MHRM-EAFC
]


def fit_pair_parameters(window_prices, idx_a, idx_b):
    log_a = np.log(window_prices[idx_a])
    log_b = np.log(window_prices[idx_b])

    beta = np.polyfit(log_b, log_a, 1)[0]
    spread = log_a - beta * log_b

    spread_mean = spread.mean()
    spread_std = spread.std()

    return beta, spread_mean, spread_std


def target_pair_position(state, beta, price_a, price_b):
    if state == 0:
        return np.array([0, 0], dtype=int)

    # Long spread = long A, short beta * B
    unit_a = 1.0
    unit_b = -beta

    # Max hedge size that keeps both legs within the $10k limit
    cap_a = DLR_LIMIT / (abs(unit_a) * price_a) if abs(unit_a) > 1e-12 else np.inf
    cap_b = DLR_LIMIT / (abs(unit_b) * price_b) if abs(unit_b) > 1e-12 else np.inf
    k = min(cap_a, cap_b)

    shares_a = int(np.round(state * k * unit_a))
    shares_b = int(np.round(state * k * unit_b))

    return np.array([shares_a, shares_b], dtype=int)


def getMyPosition(prcSoFar):
    global currentPos, pairStates, pairHoldingDays

    nins, nt = prcSoFar.shape

    if pairStates is None or pairHoldingDays is None:
        pairStates = np.zeros(len(PAIRS), dtype=int)
        pairHoldingDays = np.zeros(len(PAIRS), dtype=int)

    # Need enough history to fit the rolling window
    if nt < LOOKBACK_DAYS:
        return np.zeros(nins, dtype=int)

    # Refit each pair on the most recent window
    window_prices = prcSoFar[:, -LOOKBACK_DAYS:]
    cur_prices = prcSoFar[:, -1]

    newPos = np.zeros(nins, dtype=int)

    for pair_idx, (idx_a, idx_b) in enumerate(PAIRS):
        beta, spread_mean, spread_std = fit_pair_parameters(window_prices, idx_a, idx_b)

        # Skip unstable pairs with almost zero spread variation
        if spread_std < 1e-12:
            pairStates[pair_idx] = 0
            pairHoldingDays[pair_idx] = 0
            continue

        log_a = np.log(cur_prices[idx_a])
        log_b = np.log(cur_prices[idx_b])

        spread_now = log_a - beta * log_b
        z_now = (spread_now - spread_mean) / spread_std

        state = pairStates[pair_idx]
        next_state = state

        # Entry rules
        if state == 0:
            if z_now < -ENTRY_Z:
                next_state = 1
            elif z_now > ENTRY_Z:
                next_state = -1

        # Exit long spread
        elif state == 1:
            if z_now > -EXIT_Z:
                next_state = 0

        # Exit short spread
        elif state == -1:
            if z_now < EXIT_Z:
                next_state = 0

        # Time stop: close the trade if it has been open too long
        if state != 0 and pairHoldingDays[pair_idx] >= MAX_HOLDING_DAYS:
            next_state = 0

        if next_state == 0:
            pairHoldingDays[pair_idx] = 0
        elif state == 0 and next_state != 0:
            pairHoldingDays[pair_idx] = 1
        elif state != 0 and next_state != 0:
            pairHoldingDays[pair_idx] += 1

        pairStates[pair_idx] = next_state

        pair_target = target_pair_position(
            next_state,
            beta,
            cur_prices[idx_a],
            cur_prices[idx_b],
        )

        newPos[idx_a] += pair_target[0]
        newPos[idx_b] += pair_target[1]

    # Keep each stock inside the official non-ALGO dollar limit after combining pairs.
    pos_limits = (DLR_LIMIT / cur_prices).astype(int)
    newPos = np.clip(newPos, -pos_limits, pos_limits).astype(int)

    currentPos = newPos
    return currentPos
