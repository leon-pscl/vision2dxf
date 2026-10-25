"""Simple exception class used by the API when a base‑block cannot be loaded.
Only the ``errors`` attribute is required for the JSON error response.
"""

class BaseBlockError(Exception):
    """Raised when a base‑block cannot be loaded or fails validation.
    The API expects an ``errors`` attribute – a list of dicts describing the problem.
    """

    def __init__(self, message: str, errors: list[dict] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []
