import pytest


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization during tests (auto-run visual tests)",
    )


@pytest.fixture
def visualize_enabled(request):
    """Fixture to check if visualization is enabled via command line."""
    return request.config.getoption("--visualize")
