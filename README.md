# Ai Router SDK

Official Python SDK for [airouter.io](https://airouter.io) - Automatically get the best LLM for any request

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)

## Installation

To install the dependencies for the Ai Router SDK, use Poetry:

```sh
poetry install --no-root
```

## Usage

To get started with the Ai Router SDK, you can initialize the `AiRouter` class with your API key:

```py
from airouter.router import AiRouter
airouter = AiRouter(apikey="yourapikey_here")
response = ai_router.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, world!"}],
)
print(response)
```

## Testing

### Running Tests

You can run the tests using:

poetry run pytest

### Running Integration Tests

Integration tests are marked with `@pytest.mark.integration`. To run integration tests, use the following command:

poetry run pytest --run-integration
