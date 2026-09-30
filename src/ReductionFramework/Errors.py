class ValueErrorCode(ValueError):
    """
        A custom ValueError that includes an explicit error code.
    """
    def __init__(self, code, message):
        # Pass the message to the base ValueError class
        super().__init__(message)
        self.code = code
        self.message = message
