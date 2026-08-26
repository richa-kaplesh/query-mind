from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from core.models import ExtractedPage


class BaseExtractor(ABC):

    @abstractmethod
    def extract(self, file_path: str) -> List[ExtractedPage]:
        pass

    def validate_file(self, file_path: str) -> bool:
        path = Path(file_path)
        if path.is_dir():                                    # fix: was path.isdir()
            raise IsADirectoryError("Path is a directory, not a file")
        elif not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return True