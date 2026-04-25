"""
Layer 8: Style Reconstruction Engine
Maps PDF styles to Word document styles
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FontWeight(Enum):
    """Font weight enumeration"""
    NORMAL = 'normal'
    BOLD = 'bold'


class TextAlignment(Enum):
    """Text alignment enumeration"""
    LEFT = 'left'
    CENTER = 'center'
    RIGHT = 'right'
    JUSTIFY = 'justify'


@dataclass
class DocumentStyle:
    """Represents a document style"""
    font_name: str
    font_size: int
    font_weight: FontWeight
    italic: bool
    color_hex: str
    alignment: TextAlignment


class StyleReconstructor:
    """Reconstructs styles from PDF for Word documents"""
    
    # Default font mappings (PDF fonts -> Word fonts)
    DEFAULT_FONT_MAPPING = {
        'Times New Roman': 'Calibri',
        'Times': 'Calibri',
        'Arial': 'Calibri',
        'Helvetica': 'Calibri',
        'Courier': 'Courier New',
        'Courier New': 'Courier New',
        'Georgia': 'Georgia',
        'Verdana': 'Verdana',
        'Trebuchet': 'Trebuchet MS',
    }
    
    # Heading level determination based on font size
    HEADING_LEVEL_MAPPING = {
        # font_size: heading_level
        28: 1,
        24: 1,
        20: 2,
        18: 2,
        16: 3,
        14: 3,
        12: 4,
    }
    
    def __init__(self, config=None, logger: logging.Logger = None):
        """Initialize style reconstructor"""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.font_mapping = self._setup_font_mapping()
    
    def _setup_font_mapping(self) -> Dict[str, str]:
        """Setup font mapping"""
        mapping = self.DEFAULT_FONT_MAPPING.copy()
        
        if self.config and hasattr(self.config, 'style'):
            mapping.update(self.config.style.font_mapping)
        
        return mapping
    
    def normalize_font(self, font_name: str) -> str:
        """
        Normalize font name from PDF to Word.
        
        Args:
            font_name: Original font name from PDF
            
        Returns:
            Mapped Word-compatible font name
        """
        # Direct match
        if font_name in self.font_mapping:
            return self.font_mapping[font_name]
        
        # Partial match
        font_lower = font_name.lower()
        for pdf_font, word_font in self.font_mapping.items():
            if pdf_font.lower() in font_lower:
                return word_font
        
        # Default to configured default font
        if self.config and hasattr(self.config, 'style'):
            return self.config.style.default_font
        
        return 'Calibri'
    
    def determine_heading_level(self, font_size: float, font_weight: str = 'normal') -> Optional[int]:
        """
        Determine heading level from font size.
        
        Args:
            font_size: Font size in points
            font_weight: Font weight ('normal' or 'bold')
            
        Returns:
            Heading level (1-6) or None if not a heading
        """
        # Direct match
        for size, level in self.HEADING_LEVEL_MAPPING.items():
            if abs(font_size - size) < 1:
                return level
        
        # Heuristic: larger fonts are likely headings
        if font_size > 16:
            if font_weight == 'bold':
                if font_size > 20:
                    return 1
                else:
                    return 2
            return 3
        
        return None
    
    def rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB to hex color"""
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex to RGB color"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_style_from_block(self, text_block) -> DocumentStyle:
        """
        Create document style from text block.
        
        Args:
            text_block: Text block with formatting info
            
        Returns:
            DocumentStyle object
        """
        # Normalize font
        font_name = self.normalize_font(text_block.font_name)
        
        # Adaptive font sizing
        font_size = self._adaptive_font_size(text_block.font_size)
        
        # Font weight
        font_weight = FontWeight.BOLD if text_block.font_weight == 'bold' else FontWeight.NORMAL
        
        # Italic
        italic = text_block.is_italic
        
        # Color
        color_hex = self.rgb_to_hex(*text_block.color)
        
        # Alignment
        alignment = TextAlignment(text_block.alignment)
        
        return DocumentStyle(
            font_name=font_name,
            font_size=font_size,
            font_weight=font_weight,
            italic=italic,
            color_hex=color_hex,
            alignment=alignment
        )
    
    def _adaptive_font_size(self, original_size: float) -> int:
        """
        Adaptively adjust font size for Word.
        
        Word typically uses different scaling than PDFs
        """
        # Most PDFs use 72 DPI, Word uses 96 DPI
        # Conversion: word_size = pdf_size * (96/72)
        
        adjusted = int(original_size * 0.95)  # Slight reduction for readability
        
        # Ensure minimum and maximum
        adjusted = max(8, min(36, adjusted))
        
        return adjusted
    
    def get_paragraph_style(self, text_block) -> Dict:
        """
        Get paragraph styling from text block.
        
        Returns:
            Dictionary with paragraph styling options
        """
        style = self.create_style_from_block(text_block)
        heading_level = self.determine_heading_level(text_block.font_size, text_block.font_weight)
        
        return {
            'font_name': style.font_name,
            'font_size': style.font_size,
            'bold': style.font_weight == FontWeight.BOLD,
            'italic': style.italic,
            'color': style.color_hex,
            'alignment': style.alignment.value,
            'heading_level': heading_level,
        }
    
    def apply_style_to_paragraph(self, paragraph, style_dict: Dict) -> None:
        """
        Apply style to python-docx paragraph.
        
        Args:
            paragraph: python-docx paragraph object
            style_dict: Style dictionary from get_paragraph_style()
        """
        try:
            # Set font properties
            if paragraph.runs:
                run = paragraph.runs[0]
            else:
                run = paragraph.add_run()
            
            run.font.name = style_dict.get('font_name', 'Calibri')
            run.font.size = style_dict.get('font_size', 11) * 1000  # Twips conversion
            run.font.bold = style_dict.get('bold', False)
            run.font.italic = style_dict.get('italic', False)
            
            # Set alignment
            alignment = style_dict.get('alignment', 'left')
            if alignment == 'center':
                paragraph.alignment = 1  # CENTER
            elif alignment == 'right':
                paragraph.alignment = 2  # RIGHT
            elif alignment == 'justify':
                paragraph.alignment = 3  # JUSTIFY
            else:
                paragraph.alignment = 0  # LEFT
        
        except Exception as e:
            self.logger.warning(f"Error applying style: {e}")
