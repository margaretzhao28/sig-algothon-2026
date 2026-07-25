"""Top 11 rolling pairs ranked by 501–700 leave-one-out contribution."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
_spec = spec_from_file_location('_base', Path(__file__).with_name('pairs_12_rolling_beta.py'))
_base = module_from_spec(_spec); _spec.loader.exec_module(_base)
PAIRS = ((37, 25, 'EELT', 'CTGI'), (45, 13, 'NGTE', 'EORC'), (18, 35, 'RTTH', 'NAYO'), (10, 46, 'SMAH', 'ILVX'), (40, 7, 'ULXY', 'HETT'), (49, 50, 'MHRM', 'EAFC'), (20, 1, 'NWIG', 'AENO'), (8, 27, 'HUXZ', 'ACAC'), (33, 12, 'MTNS', 'MSDP'), (31, 43, 'ACIX', 'ITPA'), (41, 36, 'BLBT', 'FWWG'))
_base.PAIRS = PAIRS; _base.reset_state()
def reset_state(): _base.reset_state()
def getMyPosition(prcSoFar): return _base.getMyPosition(prcSoFar)
