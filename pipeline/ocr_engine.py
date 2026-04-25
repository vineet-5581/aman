"""
Layer 5: OCR Super-Pipeline
Handles scanned PDFs with preprocessing and multi-engine OCR
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import cv2
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF


@dataclass
class OCRResult:
    """OCR result for a text region."""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    engine: str  # paddleocr, tesseract


class OCREngine:
    """
    OCR Engine with preprocessing pipeline.
    Supports PaddleOCR and Tesseract with automatic fallback.
    """

    def __init__(self, logger: logging.Logger, engine: str = "paddleocr"):
        """
        Initialize OCR engine.
        
        Args:
            logger: Logger instance
            engine: Primary OCR engine (paddleocr or tesseract)
        """
        self.logger = logger
        self.engine = engine
        self.paddle_ocr = None
        self.tesseract_path = None
        self._init_engines()

    def _init_engines(self):
        """Initialize OCR engines."""
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            self.logger.info("✓ PaddleOCR initialized")
        except ImportError:
            self.logger.warning("PaddleOCR not available, will use Tesseract")
        except Exception as e:
            self.logger.warning(f"Failed to initialize PaddleOCR: {e}")

        # Initialize Tesseract
        try:
            import pytesseract
            # Detect tesseract path
            self.tesseract_path = pytesseract.pytesseract.run_and_get_output
            self.logger.info("✓ Tesseract OCR available")
        except Exception as e:
            self.logger.warning(f"Tesseract not available: {e}")

    def process_pdf(self, pdf_path: str) -> List[List[OCRResult]]:
        """
        Process PDF pages with OCR.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of pages, each containing OCRResult list
        """
        try:
            doc = fitz.open(pdf_path)
            all_results = []
            
            for page_num, page in enumerate(doc):
                # Convert page to image
                image = self._page_to_image(page)
                
                # Process image
                results = self.process_image(image)
                all_results.append(results)
                
                self.logger.debug(f"OCR processed page {page_num + 1}: {len(results)} blocks")
            
            doc.close()
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error processing PDF with OCR: {e}")
            return [[]]

    def process_image(self, image: np.ndarray) -> List[OCRResult]:
        """
        Process image with OCR.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            List of OCRResult objects
        """
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Try primary engine
        results = []
        
        if self.engine == "paddleocr" and self.paddle_ocr:
            results = self._ocr_with_paddle(processed)
        elif self.engine == "tesseract":
            results = self._ocr_with_tesseract(processed)
        
        # Fallback to alternative engine if no results
        if not results:
            self.logger.warning(f"Primary OCR engine failed, trying fallback")
            if self.engine == "paddleocr" and self.tesseract_path:
                results = self._ocr_with_tesseract(processed)
            elif self.paddle_ocr:
                results = self._ocr_with_paddle(processed)
        
        return results

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR.
        
        Steps:
        1. Deskew
        2. Denoise
        3. Binarization
        4. Contrast enhancement
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Deskew
            gray = self._deskew(gray)
            
            # Denoise
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Binarization
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {e}")
            return image

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew image."""
        try:
            # Get coordinates of non-white pixels
            coords = np.column_stack(np.where(image < 225))
            
            if len(coords) == 0:
                return image
            
            # Fit line to get angle
            angle = cv2.minAreaRect(cv2.convexHull(coords))[-1]
            
            # Correct angle
            if angle < -45:
                angle = 90 + angle
            
            # Apply rotation if needed
            if abs(angle) > 0.5:
                h, w = image.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), 
                                      borderMode=cv2.BORDER_REPLICATE)
            
            return image
        except Exception as e:
            self.logger.warning(f"Deskew failed: {e}")
            return image

    def _ocr_with_paddle(self, image: np.ndarray) -> List[OCRResult]:
        """OCR using PaddleOCR."""
        try:
            if not self.paddle_ocr:
                return []
            
            results = self.paddle_ocr.ocr(image, cls=True)
            
            ocr_results = []
            if results and results[0]:
                for line in results[0]:
                    bbox = line[0]  # List of 4 corner points
                    text = line[1][0]
                    confidence = float(line[1][1])
                    
                    # Convert bbox format
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    bbox_rect = (min(x_coords), min(y_coords), 
                               max(x_coords), max(y_coords))
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=bbox_rect,
                        engine="paddleocr"
                    ))
            
            return ocr_results
            
        except Exception as e:
            self.logger.error(f"PaddleOCR error: {e}")
            return []

    def _ocr_with_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """OCR using Tesseract."""
        try:
            import pytesseract
            
            # Get detailed data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            ocr_results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if text and confidence > 0:
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence / 100.0,
                        bbox=bbox,
                        engine="tesseract"
                    ))
            
            return ocr_results
            
        except Exception as e:
            self.logger.error(f"Tesseract error: {e}")
            return []

    def _page_to_image(self, page: fitz.Page, dpi: int = 300) -> np.ndarray:
        """Convert PDF page to image."""
        # Render at high DPI for better OCR
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = pix.tobytes("ppm")
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
