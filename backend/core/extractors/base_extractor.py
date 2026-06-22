from abc import ABC, abstractmethod
from core.models import ExtractedPage

class BaseExtractor(ABC):
    
    @abstractmethod
    def extract(self, file_path: str) -> list[ExtractedPage]:
        pass
    
    def validate_file(self, file_path: str) -> bool:
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        return True