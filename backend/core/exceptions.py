class DocumentExtractionError(Exception):
    """Base class for extraction failures that need a specific, user-facing reason."""


class PDFPasswordProtectedError(DocumentExtractionError):
    """Raised when a PDF requires a password fitz cannot supply."""


class PDFCorruptError(DocumentExtractionError):
    """Raised when a PDF fails to open due to malformed/corrupt file data."""


class GatewayError(Exception):
    """An LLM Gateway call failed.

    status: HTTP status from the gateway, or None when there was no response at all.
    reason: "http" (gateway answered with an error status), "unreachable" (timeout or
            connection failure) or "budget" (this question used up its call budget).
    """

    def __init__(self, status: int | None, message: str, reason: str = "http"):
        super().__init__(message)
        self.status = status
        self.reason = reason

    @property
    def fatal(self) -> bool:
        """True when sending more requests cannot help: rate limited (429), server error
        (5xx), no response at all, or call budget spent. Other 4xx (e.g. one malformed
        request) are not fatal: the forced-answer recovery path can still work."""
        return self.status is None or self.status == 429 or self.status >= 500