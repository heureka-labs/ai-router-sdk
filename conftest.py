import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", help="run integration tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        # Skip "expensive" marked tests if --runexpensive is not set
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Skipping integration tests, as they need the router to be accessible locally. Use --run-integration to run."
                    )
                )
