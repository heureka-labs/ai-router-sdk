class SDKError(Exception):
    """Base class for all SDK exceptions."""

    pass


class MissingAPIKeyError(SDKError):
    """Exception raised when the API key is not provided."""

    def __init__(self, message="API key is required but not provided."):
        super().__init__(message)


class MissingMessagesError(SDKError):
    """Exception raised when messages are not provided."""

    def __init__(self, message="Messages are required but not provided."):
        super().__init__(message)


class StreamingNotSupportedInPrivacyModeError(SDKError):
    """Exception raised when streaming is enabled in privacy mode."""

    def __init__(self, message="Streaming is not supported in privacy mode."):
        super().__init__(message)
