from PIL import Image
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageExtractor:

    def __init__(self):

        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def extract_text(self, image_path: str) -> Dict[str, any]:
 
        image_path = Path(image_path)
        
        # Validation
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {image_path.suffix}")
        
        logger.info(f"Starting OCR on: {image_path.name}")
        
        try:

            image = Image.open(image_path)
            logger.info(f" Image size: {image.size[0]}x{image.size[1]} pixels")

            if image.mode != 'RGB':
                logger.info(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')

            text = self._run_ocr(image)
            
            return {
                'text': text,
                'source_file': image_path.name,
                'extraction_method': 'ocr',
                'image_size': f"{image.size[0]}x{image.size[1]}",
                'char_count': len(text),
                'success': len(text) > 0
            }
        
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return {
                'text': '',
                'source_file': image_path.name,
                'extraction_method': 'ocr',
                'success': False,
                'error': str(e)
            }
    
    def _run_ocr(self, image: Image.Image) -> str:

        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='en',
            show_log=False
        )
        

        result = ocr.ocr(image, cls=True)

        if not result or not result[0]:
            logger.warning(" OCR found no text in image")
            return ""
        
        extracted_lines = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]

            if confidence > 0.6:
                extracted_lines.append(text)
            else:
                logger.debug(f"Skipped low confidence: {text} ({confidence:.2f})")
        
        logger.info(f" Extracted {len(extracted_lines)} lines from image")
        return "\n".join(extracted_lines)


if __name__ == "__main__":
    extractor = ImageExtractor()
    result = extractor.extract_text("menu_photo.jpg")
    
    print(f"\n Results:")
    print(f"Success: {result['success']}")
    print(f"Characters: {result['char_count']}")
    print(f"\n Text:\n{result['text'][:500]}")
