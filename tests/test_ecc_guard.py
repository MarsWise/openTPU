"""Hardware test gate must not accept missing or dirty ECC evidence."""
import pytest

from opentpu.host.ecc_guard import check_board_ecc, require_clean_ecc
from opentpu.host.regs import ST_ECC_DED


def test_clean_two_channels():
    require_clean_ecc([(0, 0), (0, 0)], 0)


@pytest.mark.parametrize('counters,status', [
    (None, 0), ([], 0), ([(0, 0)], 0), ([(0, 0), None], 0),
    ([(0, 0), (0,)], 0), ([(0, 0), (0, None)], 0),
    ([(0, 0), (0, -1)], 0), ([(1, 0), (0, 0)], 0),
    ([(0, 0), (0, 1)], 0), ([(0, 0), (0, 0)], None),
    ([(0, 0), (0, 0)], ST_ECC_DED),
])
def test_reject_missing_or_dirty_ecc(counters, status):
    with pytest.raises(ValueError):
        require_clean_ecc(counters, status)


def test_snapshot_detects_status_change_during_read(monkeypatch):
    pytest.importorskip('fcntl', reason='board driver runs on Linux')
    from opentpu.host import memcal

    class Board:
        t = object()
        statuses = iter([0, ST_ECC_DED])

        def info(self):
            return {'status': next(self.statuses)}

    monkeypatch.setattr(memcal, 'ecc', lambda t: [(0, 0), (0, 0)])
    with pytest.raises(ValueError, match='ECC_DED'):
        check_board_ecc(Board())


def test_backend_closes_before_scrub_if_ecc_is_dirty(monkeypatch):
    pytest.importorskip('fcntl', reason='board driver runs on Linux')
    from opentpu.host import board as board_module, ecc_guard

    class FakeBoard:
        closed = False
        scrubbed = False

        def __init__(self, transport):
            pass

        def close(self):
            self.closed = True

        def scrub(self):
            self.scrubbed = True

    fake = FakeBoard(None)
    monkeypatch.setattr(board_module, 'Board', lambda transport: fake)
    def fail(_):
        raise ValueError('dirty ECC')
    monkeypatch.setattr(ecc_guard, 'check_board_ecc', fail)
    with pytest.raises(ValueError, match='dirty ECC'):
        board_module.BoardBackend(None, [], guard_ecc=True)
    assert fake.closed and not fake.scrubbed
