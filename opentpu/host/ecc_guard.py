"""Fail-closed ECC evidence for hardware validation runs.

This only interprets snapshots. It does not clear counters or alter the board.
"""
from numbers import Integral

from .regs import ST_ECC_DED


def require_clean_ecc(counters, status):
    """Reject unavailable, malformed, corrected or uncorrectable ECC evidence."""
    if not isinstance(status, Integral) or status < 0:
        raise ValueError('ECC status unavailable or malformed')
    if status & ST_ECC_DED:
        raise ValueError('ECC_DED is set')
    if not isinstance(counters, (tuple, list)) or len(counters) != 2:
        raise ValueError('ECC counters unavailable or malformed')
    for channel in counters:
        if not isinstance(channel, (tuple, list)) or len(channel) != 2:
            raise ValueError('ECC channel counters unavailable or malformed')
        if any(not isinstance(value, Integral) or value < 0 for value in channel):
            raise ValueError('ECC counter unavailable or malformed')
        if any(channel):
            raise ValueError('ECC counter is nonzero')


def check_board_ecc(board):
    """Take a snapshot before a potentially clearing initialization, while locked."""
    from . import memcal

    status_before = board.info()['status']
    counters = memcal.ecc(board.t)
    status_after = board.info()['status']
    require_clean_ecc(counters, status_before)
    require_clean_ecc(counters, status_after)
    return {'counters': counters, 'status_before': status_before, 'status_after': status_after}
