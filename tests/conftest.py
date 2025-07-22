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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--visualize"):
        # --visualize given: do not skip visualize tests
        return
    skip_visualize = pytest.mark.skip(reason="need --visualize option to run this test")
    for item in items:
        if "visualize" in item.keywords:
            item.add_marker(skip_visualize)
