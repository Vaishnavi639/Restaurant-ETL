from pathlib import Path
from typing import Dict
import logging
from .pdf_extractor import PDFExtractor
from .image_extractor import ImageExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalExtractor:
    def __init__(self):
  
        self.pdf_extractor = PDFExtractor()
        self.image_extractor = ImageExtractor()
        self.extractor_map = {
            '.pdf': self.pdf_extractor,
            '.jpg': self.image_extractor,
            '.jpeg': self.image_extractor,
            '.png': self.image_extractor,
            '.bmp': self.image_extractor,
            '.tiff': self.image_extractor,
            '.tif': self.image_extractor,
        }
    
    def extract(self, file_path: str) -> Dict[str, any]:
       
        file_path = Path(file_path)
        
    
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
  
        extension = file_path.suffix.lower()
        
      
        extractor = self.extractor_map.get(extension)
        
        if not extractor:
            raise ValueError(
                f"Unsupported file type: {extension}\n"
                f"Supported: {list(self.extractor_map.keys())}"
            )
        
        logger.info(f"Using {extractor.__class__.__name__} for {file_path.name}")

        return extractor.extract_text(str(file_path))
    
    def get_supported_formats(self) -> list:
        """Return list of supported file formats."""
        return list(self.extractor_map.keys())


if __name__ == "__main__":
    extractor = UniversalExtractor()
    
    print(f" Supported formats: {extractor.get_supported_formats()}")
    
    # Auto-detects file type and extracts
    result = extractor.extract("menu.pdf")  # or menu.jpg, menu.png, etc.
    print(f"\n Extracted {result['char_count']} characters")
