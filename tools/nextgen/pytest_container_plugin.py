"""pytest plugin for small containers (-p pytest_container_plugin with tools/nextgen on PYTHONPATH): the caches' 20 GB disk floor lowered to 2 GB, Verilator builds one job at a time (the S=2 m8 testbench OOMs at -j 2 in 7 GB)."""
import pytest

@pytest.fixture(autouse=True)
def _low_floor(monkeypatch):
    import opentpu.progcache as P, opentpu.qcache as Q, opentpu.imgcache as G
    for m in (P, Q, G):
        monkeypatch.setattr(m, "FREE_FLOOR", 2 << 30)


def pytest_configure(config):
    """7 GB container: the S=2 m8 testbench's C++ OOMs at -j 2; build it one job at a time."""
    import opentpu.rtlsim as R
    i = R.VFLAGS.index("-j")
    R.VFLAGS[i + 1] = "1"
