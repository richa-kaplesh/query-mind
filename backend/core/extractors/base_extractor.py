from abc import ABC, abstractmethod
from core.models import ExtractedPage
from pathlib import Path

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, file_path:str)->List[ExtractedPage]:
        pass

    def validate_file(self,file_path:str)->bool:
       path = Path(file_path)
       if path.isdir():
            raise IsADirectoryError("Path is a directory, not a file")
       elif not path.exists():
           raise FileNotFoundError("File not found")
       else:
              return True