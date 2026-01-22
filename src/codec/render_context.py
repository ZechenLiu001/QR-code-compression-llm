"""Render context codec - text rendering to PNG"""

import os
from pathlib import Path
from typing import Union, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from .base import BaseCodec


class RenderCodec(BaseCodec):
    """Text rendering to PNG with built-in font
    
    Uses DejaVu Sans Mono font (OFL license) for cross-machine consistency
    """
    
    def __init__(self, font_size: int = 14, line_width: int = 80, dpi: int = 150):
        """Initialize render codec
        
        Args:
            font_size: Font size in points
            line_width: Characters per line
            dpi: DPI for rendering
        """
        self.font_size = font_size
        self.line_width = line_width
        self.dpi = dpi
        self.font = self._load_builtin_font(font_size)
    
    def _load_builtin_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load built-in font
        
        Args:
            size: Font size
            
        Returns:
            PIL Font object
        """
        # Try to load built-in font
        font_path = Path(__file__).parent.parent.parent / "assets" / "fonts" / "DejaVuSansMono.ttf"
        
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                pass
        
        # Fallback to default monospace font
        try:
            return ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            # Final fallback
            return ImageFont.load_default()
    
    def encode(self, text: str) -> Image.Image:
        """Render text to PNG image
        
        Args:
            text: Input text
            
        Returns:
            PIL Image (grayscale, black text on white background)
        """
        # Split text into lines
        lines = []
        words = text.split()
        current_line = []
        current_width = 0
        
        for word in words:
            word_len = len(word)
            if current_width + word_len + 1 <= self.line_width:
                current_line.append(word)
                current_width += word_len + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_len
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Calculate image dimensions
        line_height = int(self.font_size * 1.2)
        padding = 20
        img_width = int(self.line_width * self.font_size * 0.6) + 2 * padding
        img_height = len(lines) * line_height + 2 * padding
        
        # Create image (grayscale, white background)
        img = Image.new("L", (img_width, img_height), color=255)
        draw = ImageDraw.Draw(img)
        
        # Draw text
        y = padding
        for line in lines:
            draw.text((padding, y), line, font=self.font, fill=0)
            y += line_height
        
        return img
    
    def decode(self, encoded: Image.Image) -> str:
        """Decode rendered image back to text (OCR-like, simplified)
        
        Note: This is a placeholder. Full OCR would require OCR library.
        For now, we assume the image can be read back (which is not realistic
        without OCR, but kept for interface consistency).
        
        Args:
            encoded: PIL Image
            
        Returns:
            Decoded text (placeholder implementation)
        """
        # This would require OCR in practice
        # For now, return empty string as placeholder
        return ""
    
    def get_token_cost(
        self,
        encoded: Image.Image,
        processor=None,
    ) -> Dict[str, Any]:
        """Get token cost for rendered image
        
        Args:
            encoded: PIL Image
            processor: Optional processor for visual token counting
            
        Returns:
            Token cost dictionary
        """
        from ..model.tokenizer_utils import get_visual_token_count
        
        w, h = encoded.size
        
        # Try to get visual tokens from processor
        visual_info = {"visual_tokens": 0, "source": "proxy", "patch_count": 0}
        if processor is not None:
            # Create dummy processor output for probing
            proc_output = {}
            visual_info = get_visual_token_count(proc_output, encoded, patch_size=14)
        
        # Fallback to proxy
        if visual_info["visual_tokens"] == 0:
            patch_size = 14
            patch_count = ((h + patch_size - 1) // patch_size) * ((w + patch_size - 1) // patch_size)
            visual_info = {
                "visual_tokens": patch_count,
                "source": "proxy",
                "patch_count": patch_count,
            }
        
        return {
            "text_tokens": 0,  # No text tokens in image mode
            "visual_tokens": visual_info["visual_tokens"],
            "visual_tokens_source": visual_info["source"],
            "patch_count": visual_info["patch_count"],
            "image_size": (w, h),
            "total_tokens": visual_info["visual_tokens"],
        }
