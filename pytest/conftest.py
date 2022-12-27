import pytest

def pytest_addoption(parser):
    parser.addoption('--arg', action='store')

@pytest.fixture
def arg(request):
    return request.config.getoption('--arg')
