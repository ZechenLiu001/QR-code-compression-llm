"""Codebook external codec - standard 2D barcode baseline"""

import qrcode
from typing import Union, List, Dict, Any
from PIL import Image
from .base import BaseCodec

try:
    from pylibdmtx import encode as dmtx_encode
    from pylibdmtx import decode as dmtx_decode
    HAS_DATAMATRIX = True
except ImportError:
    HAS_DATAMATRIX = False

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


class CodebookExternalCodec(BaseCodec):
    """Standard 2D barcode baseline (external encode/decode)
    
    Supports:
    - QR Code: qrcode library for encoding
    - DataMatrix: pylibdmtx for encoding/decoding
    
    Purpose:
    - Baseline comparison for custom codebook
    - External decode upper bound (not model-dependent)
    - Long text support via chunking
    """
    
    def __init__(
        self,
        format: str = "qrcode",
        error_correction: str = "L",
        max_bytes_per_symbol: int = 1200,
    ):
        """Initialize external codec
        
        Args:
            format: "qrcode" or "datamatrix"
            error_correction: Error correction level (QR: L/M/Q/H)
            max_bytes_per_symbol: Maximum bytes per symbol
        """
        self.format = format
        self.error_correction = error_correction
        self.max_bytes_per_symbol = max_bytes_per_symbol
        
        # Map error correction
        if format == "qrcode":
            ec_map = {"L": qrcode.constants.ERROR_CORRECT_L,
                     "M": qrcode.constants.ERROR_CORRECT_M,
                     "Q": qrcode.constants.ERROR_CORRECT_Q,
                     "H": qrcode.constants.ERROR_CORRECT_H}
            self.ec_level = ec_map.get(error_correction, qrcode.constants.ERROR_CORRECT_L)
    
    def encode(self, text: str) -> Union[Image.Image, List[Image.Image]]:
        """Encode text to 2D barcode(s)
        
        Long text: chunk into multiple symbols
        
        Args:
            text: Input text
            
        Returns:
            Single image or list of images
        """
        text_bytes = text.encode('utf-8')
        
        if len(text_bytes) <= self.max_bytes_per_symbol:
            # Single symbol
            return self._encode_single(text_bytes)
        else:
            # Multiple symbols
            images = []
            chunks = self._chunk_bytes(text_bytes, self.max_bytes_per_symbol)
            for chunk in chunks:
                img = self._encode_single(chunk)
                images.append(img)
            return images
    
    def _encode_single(self, data: bytes) -> Image.Image:
        """Encode single symbol"""
        if self.format == "qrcode":
            qr = qrcode.QRCode(
                version=None,  # Auto
                error_correction=self.ec_level,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            return img.convert('L')  # Grayscale
        
        elif self.format == "datamatrix":
            if not HAS_DATAMATRIX:
                raise ImportError("pylibdmtx not available for DataMatrix encoding")
            
            encoded = dmtx_encode(data)
            # Convert to PIL Image
            # Note: pylibdmtx returns raw bytes, need to convert
            # This is simplified - actual implementation may vary
            raise NotImplementedError("DataMatrix encoding needs proper image conversion")
        
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _chunk_bytes(self, data: bytes, chunk_size: int) -> List[bytes]:
        """Chunk bytes into smaller pieces"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
    
    def decode(self, encoded: Union[Image.Image, List[Image.Image]]) -> str:
        """Decode 2D barcode(s) back to text using external library
        
        Args:
            encoded: Single image or list of images
            
        Returns:
            Decoded text string
        """
        if isinstance(encoded, list):
            # Decode multiple symbols and concatenate
            texts = []
            for img in encoded:
                text = self._decode_single(img)
                texts.append(text)
            return "".join(texts)
        else:
            return self._decode_single(encoded)
    
    def _decode_single(self, img: Image.Image) -> str:
        """Decode single symbol"""
        if self.format == "qrcode":
            if HAS_PYZBAR:
                # Use pyzbar for decoding
                from pyzbar import pyzbar
                decoded = pyzbar.decode(img)
                if decoded:
                    return decoded[0].data.decode('utf-8')
                else:
                    raise ValueError("Failed to decode QR code")
            else:
                # Fallback: would need to use VLM or other method
                # For now, raise error
                raise NotImplementedError("QR decoding requires pyzbar or VLM")
        
        elif self.format == "datamatrix":
            if not HAS_DATAMATRIX:
                raise ImportError("pylibdmtx not available for DataMatrix decoding")
            
            decoded = dmtx_decode(img)
            if decoded:
                return decoded[0].data.decode('utf-8')
            else:
                raise ValueError("Failed to decode DataMatrix")
        
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def get_token_cost(
        self,
        encoded: Union[Image.Image, List[Image.Image]],
        processor=None,
    ) -> Dict[str, Any]:
        """Get token cost for external codec image(s)"""
        from ..model.tokenizer_utils import get_visual_token_count
        
        if isinstance(encoded, list):
            # Sum tokens from all images
            total_visual = 0
            total_patches = 0
            source = None
            
            for img in encoded:
                proc_output = {}
                visual_info = get_visual_token_count(proc_output, img, patch_size=14)
                total_visual += visual_info["visual_tokens"]
                total_patches += visual_info["patch_count"]
                if source is None:
                    source = visual_info["source"]
            
            # Use first image size as representative
            w, h = encoded[0].size if encoded else (0, 0)
            
            return {
                "text_tokens": 0,
                "visual_tokens": total_visual,
                "visual_tokens_source": source or "proxy",
                "patch_count": total_patches,
                "image_size": (w, h),
                "total_tokens": total_visual,
            }
        else:
            proc_output = {}
            visual_info = get_visual_token_count(proc_output, encoded, patch_size=14)
            
            w, h = encoded.size
            
            return {
                "text_tokens": 0,
                "visual_tokens": visual_info["visual_tokens"],
                "visual_tokens_source": visual_info["source"],
                "patch_count": visual_info["patch_count"],
                "image_size": (w, h),
                "total_tokens": visual_info["visual_tokens"],
            }
