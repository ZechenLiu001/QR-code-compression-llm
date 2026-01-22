"""Codebook context codec - 2D codebook v0 with zlib compression"""

import zlib
import struct
import numpy as np
from typing import Union, List, Dict, Any
from PIL import Image
from .base import BaseCodec


class CodebookCodec(BaseCodec):
    """2D Codebook v0 implementation
    
    Structure:
    - Header (24 bytes, written twice for robustness)
    - Finder Patterns (7x7 modules at four corners)
    - Payload (compressed bytes as bitstream)
    """
    
    # Header structure
    HEADER_SIZE = 24  # bytes
    MAGIC = b"CBK0"
    
    def __init__(self, color_mode: str = "4color", cell_size: int = 4, max_grid: int = 192):
        """Initialize codebook codec
        
        Args:
            color_mode: "bw" (1bpp) or "4color" (2bpp)
            cell_size: Pixel size per cell
            max_grid: Maximum grid size (cells per side)
        """
        self.color_mode = color_mode
        self.cell_size = cell_size
        self.bits_per_cell = 1 if color_mode == "bw" else 2
        self.max_grid = max_grid
        self.finder_size = 7  # Finder pattern size
    
    def encode(self, text: str) -> Image.Image:
        """Encode text to codebook image
        
        Args:
            text: Input text
            
        Returns:
            PIL Image (codebook)
        """
        # Compress text
        raw_bytes = text.encode('utf-8')
        payload_bytes = zlib.compress(raw_bytes, level=6)
        
        # Calculate grid dimensions
        payload_bits = len(payload_bytes) * 8
        cells_needed = (payload_bits + self.bits_per_cell - 1) // self.bits_per_cell
        
        # Add header space (written twice) and finder patterns
        header_cells = (self.HEADER_SIZE * 8 + self.bits_per_cell - 1) // self.bits_per_cell
        finder_cells = self.finder_size * 2  # Top and bottom finders
        
        total_cells_per_side = int(np.ceil(np.sqrt(cells_needed + header_cells * 2 + finder_cells * 2)))
        total_cells_per_side = min(total_cells_per_side, self.max_grid)
        
        # Create grid
        grid = np.zeros((total_cells_per_side, total_cells_per_side), dtype=np.uint8)
        
        # Place finder patterns at corners
        self._place_finder_patterns(grid, total_cells_per_side)
        
        # Encode header (write twice for robustness)
        header_data = self._encode_header(payload_bytes, chunk_id=0, total_chunks=1)
        header_bits = self._bytes_to_bits(header_data)
        self._write_bits_to_grid(grid, header_bits, self.finder_size, 0, self.bits_per_cell)
        # Write header second time
        header_start2 = self.finder_size + len(header_bits) // self.bits_per_cell
        self._write_bits_to_grid(grid, header_bits, header_start2, 0, self.bits_per_cell)
        
        # Encode payload
        payload_bits = self._bytes_to_bits(payload_bytes)
        payload_start = self.finder_size + (len(header_bits) // self.bits_per_cell) * 2
        self._write_bits_to_grid(grid, payload_bits, payload_start, 0, self.bits_per_cell)
        
        # Convert grid to image
        img = self._grid_to_image(grid)
        
        return img
    
    def decode(self, encoded_img: Image.Image) -> str:
        """Decode codebook image back to text
        
        Note: This is a simplified decoder. Full implementation would require
        robust finder pattern detection and error correction.
        
        Args:
            encoded_img: PIL Image (codebook)
            
        Returns:
            Decoded text string
        """
        # Convert image to grid
        grid = self._image_to_grid(encoded_img)
        
        # Detect finder patterns and extract header
        # Simplified: assume standard layout
        finder_size = self.finder_size
        header_start = finder_size
        
        # Read header bits
        header_bits = self._read_bits_from_grid(
            grid, header_start, 0, self.HEADER_SIZE * 8, self.bits_per_cell
        )
        header_bytes = self._bits_to_bytes(header_bits)
        
        # Parse header
        magic = header_bytes[:4]
        if magic != self.MAGIC:
            raise ValueError("Invalid codebook magic")
        
        version = header_bytes[4]
        flags = header_bytes[5]
        payload_len = struct.unpack('>I', header_bytes[10:14])[0]
        crc32 = struct.unpack('>I', header_bytes[14:18])[0]
        
        # Read payload
        payload_start = finder_size + (self.HEADER_SIZE * 8 + self.bits_per_cell - 1) // self.bits_per_cell
        payload_bits = self._read_bits_from_grid(
            grid, payload_start, 0, payload_len * 8, self.bits_per_cell
        )
        payload_bytes = self._bits_to_bytes(payload_bits)
        
        # Verify CRC32
        computed_crc = zlib.crc32(payload_bytes) & 0xffffffff
        if computed_crc != crc32:
            raise ValueError(f"CRC32 mismatch: {computed_crc} != {crc32}")
        
        # Decompress
        raw_bytes = zlib.decompress(payload_bytes)
        text = raw_bytes.decode('utf-8')
        
        return text
    
    def _encode_header(self, payload_bytes: bytes, chunk_id: int, total_chunks: int) -> bytes:
        """Encode header
        
        Args:
            payload_bytes: Compressed payload bytes
            chunk_id: Chunk identifier
            total_chunks: Total number of chunks
            
        Returns:
            Header bytes
        """
        flags = 0
        if self.color_mode == "4color":
            flags |= 0x01
        flags |= 0x04  # compression flag (zlib)
        
        payload_len = len(payload_bytes)
        crc32 = zlib.crc32(payload_bytes) & 0xffffffff
        
        header = struct.pack(
            '>4sBBHHIIHH',
            self.MAGIC,
            0,  # version
            flags,
            chunk_id,
            total_chunks,
            payload_len,
            crc32,
            0,  # header_crc16 (placeholder)
            0,  # reserved
        )
        
        return header
    
    def _create_finder_pattern(self) -> np.ndarray:
        """Create 7x7 finder pattern (QR-like)"""
        pattern = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.uint8)
        return pattern
    
    def _place_finder_patterns(self, grid: np.ndarray, size: int):
        """Place finder patterns at four corners"""
        pattern = self._create_finder_pattern()
        finder_size = self.finder_size
        
        # Top-left
        grid[:finder_size, :finder_size] = pattern
        # Top-right
        grid[:finder_size, size-finder_size:] = pattern
        # Bottom-left
        grid[size-finder_size:, :finder_size] = pattern
        # Bottom-right
        grid[size-finder_size:, size-finder_size:] = pattern
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to list of bits"""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes"""
        bytes_list = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= (bits[i + j] << (7 - j))
            bytes_list.append(byte)
        return bytes(bytes_list)
    
    def _write_bits_to_grid(
        self, grid: np.ndarray, bits: List[int], start_row: int, start_col: int, bits_per_cell: int
    ):
        """Write bits to grid"""
        rows, cols = grid.shape
        bit_idx = 0
        
        for row in range(start_row, rows):
            for col in range(start_col, cols):
                if bit_idx >= len(bits):
                    return
                
                # Skip finder pattern areas
                if self._is_in_finder_area(row, col, rows, cols):
                    continue
                
                # Write bits
                if bits_per_cell == 1:
                    grid[row, col] = bits[bit_idx]
                    bit_idx += 1
                else:  # 2 bits per cell
                    if bit_idx + 1 < len(bits):
                        value = (bits[bit_idx] << 1) | bits[bit_idx + 1]
                        grid[row, col] = value
                        bit_idx += 2
                    else:
                        # Pad with 0
                        grid[row, col] = bits[bit_idx] << 1
                        bit_idx += 1
    
    def _read_bits_from_grid(
        self, grid: np.ndarray, start_row: int, start_col: int, num_bits: int, bits_per_cell: int
    ) -> List[int]:
        """Read bits from grid"""
        rows, cols = grid.shape
        bits = []
        bit_idx = 0
        
        for row in range(start_row, rows):
            for col in range(start_col, cols):
                if bit_idx >= num_bits:
                    break
                
                # Skip finder pattern areas
                if self._is_in_finder_area(row, col, rows, cols):
                    continue
                
                # Read bits
                if bits_per_cell == 1:
                    bits.append(int(grid[row, col]))
                    bit_idx += 1
                else:  # 2 bits per cell
                    value = grid[row, col]
                    bits.append((value >> 1) & 1)
                    bits.append(value & 1)
                    bit_idx += 2
                    if bit_idx >= num_bits:
                        break
        
        return bits[:num_bits]
    
    def _is_in_finder_area(self, row: int, col: int, rows: int, cols: int) -> bool:
        """Check if position is in finder pattern area"""
        finder_size = self.finder_size
        return (
            (row < finder_size and col < finder_size) or
            (row < finder_size and col >= cols - finder_size) or
            (row >= rows - finder_size and col < finder_size) or
            (row >= rows - finder_size and col >= cols - finder_size)
        )
    
    def _grid_to_image(self, grid: np.ndarray) -> Image.Image:
        """Convert grid to PIL Image"""
        # Scale grid by cell_size
        h, w = grid.shape
        img_array = np.repeat(np.repeat(grid, self.cell_size, axis=0), self.cell_size, axis=1)
        
        # Convert to image
        if self.color_mode == "bw":
            # Grayscale: 0 -> white (255), 1 -> black (0)
            img_array = 255 - (img_array * 255)
            img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        else:  # 4color
            # Map 0-3 to colors: white, light gray, dark gray, black
            color_map = np.array([255, 170, 85, 0], dtype=np.uint8)
            img_array = color_map[img_array]
            img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        
        return img
    
    def _image_to_grid(self, img: Image.Image) -> np.ndarray:
        """Convert PIL Image to grid"""
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img)
        
        # Downsample to grid
        h, w = img_array.shape
        grid_h = h // self.cell_size
        grid_w = w // self.cell_size
        
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        for i in range(grid_h):
            for j in range(grid_w):
                # Sample center of cell
                y = i * self.cell_size + self.cell_size // 2
                x = j * self.cell_size + self.cell_size // 2
                if y < h and x < w:
                    value = img_array[y, x]
                    if self.color_mode == "bw":
                        grid[i, j] = 0 if value < 128 else 1
                    else:  # 4color
                        grid[i, j] = min(3, value // 64)
        
        return grid
    
    def get_token_cost(
        self,
        encoded: Image.Image,
        processor=None,
    ) -> Dict[str, Any]:
        """Get token cost for codebook image"""
        from ..model.tokenizer_utils import get_visual_token_count
        
        w, h = encoded.size
        
        # Try to get visual tokens from processor
        visual_info = {"visual_tokens": 0, "source": "proxy", "patch_count": 0}
        if processor is not None:
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
            "text_tokens": 0,
            "visual_tokens": visual_info["visual_tokens"],
            "visual_tokens_source": visual_info["source"],
            "patch_count": visual_info["patch_count"],
            "image_size": (w, h),
            "total_tokens": visual_info["visual_tokens"],
        }
