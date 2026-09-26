class DocumentExtractionError(Exception):
    """Base class for extraction failures that need a specific, user-facing reason."""


class PDFPasswordProtectedError(DocumentExtractionError):
    """Raised when a PDF requires a password fitz cannot supply."""


class PDFCorruptError(DocumentExtractionError):
    """Raised when a PDF fails to open due to malformed/corrupt file data."""