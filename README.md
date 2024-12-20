<div align="center">

<a href="https://airouter.io" target="_blank">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://airouter.io/assets/images/logo-w.png" style="max-width: 100%; width: 200px; margin-bottom: 20px">
        <img alt="AI Router Logo" src="https://airouter.io/assets/images/logo.png" width="200px">
    </picture>
</a>

#

🪄 **AI Router**: Automatically get the best LLM for any request.

<h4>

[Documentation](https://airouter.io/docs) | [Pricing](https://airouter.io/pricing) | [FAQ](https://airouter.io/faq)

</h4>

[![PyPI version](https://badge.fury.io/py/airouter-sdk.svg)](https://badge.fury.io/py/airouter-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/airouter-sdk.svg)](https://pypi.org/project/airouter-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


## Installation

### Basic Installation

Install the package using pip:

```bash
pip install "airouter_sdk"
```

### Privacy Features

To enable privacy mode for local embedding generation:

```bash
pip install "airouter_sdk[privacy]"
```

### Development Setup

For SDK development, use Poetry:

```bash
poetry install --no-root
```

## Usage

### Basic Usage

Initialize the `AiRouter` class with your API key:

```python
from airouter.router import AiRouter

# Initialize with API key
airouter = AiRouter(api_key="your_api_key_here")  # Note: Use AIROUTER_API_KEY env var instead in production

# Create a chat completion
response = airouter.chat.completions.create(
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)
print(response)
```

### Environment Variables

For security best practices, set your API key as an environment variable:

```bash
export AIROUTER_API_KEY="your_api_key_here"
```

### Model Selection

You can very easily retrieve the best model:

```python
# Get the best model for your use case
model_name = airouter.get_best_model(
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)
print(model_name)  # Returns the recommended model identifier
```

This is particularly useful when working with private models or when you need to handle the model calling logic yourself.

You can also limit the models, for example to create a simple switch between gpt-4o-mini and gpt-4o:

```python
from airouter import AiRouter, Model

model_name = airouter.get_best_model(
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    models=[Model.GPT_4O_MINI, Model.GPT_4O]
)
if model_name == Model.GPT_4O_MINI:
    # call your private gpt-4o-mini endpoint
else:
    # call your private gpt-4o endpoint
```

### Privacy Features

For enhanced privacy, use the `full_privacy` mode to avoid sending message content:

```python
# Use privacy mode for sensitive content
model_name = airouter.get_best_model(
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    full_privacy=True
)
print(model_name)
```

This mode:
- Generates embeddings locally
- Only sends embeddings to AI Router
- Protects sensitive message content
- Requires the privacy dependencies: `pip install "airouter_sdk[privacy]"`

## Testing

### Running Tests

Run the test suite using Poetry:

```bash
poetry run pytest
```

### Integration Tests

For running integration tests (marked with `@pytest.mark.integration`):

```bash
poetry run pytest --run-integration
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
