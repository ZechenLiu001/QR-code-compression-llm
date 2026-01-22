"""Smoke test for basic functionality"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.json_task import generate_json_sample
from src.data.needle_task import generate_needle_sample
from src.codec.text_context import TextCodec
from src.codec.render_context import RenderCodec
from src.codec.codebook_context import CodebookCodec
from src.codec.codebook_external import CodebookExternalCodec
from src.eval.metrics import json_exact_match, json_structural_f1, needle_hit_at_1
from src.utils.seed import set_seed


def test_data_generation():
    """Test data generation"""
    print("Testing data generation...")
    
    set_seed(42)
    
    # Test JSON generation
    json_sample = generate_json_sample(1024, field_count=5, seed=42)
    assert "context" in json_sample
    assert "query" in json_sample
    assert "answer" in json_sample
    print("✓ JSON generation works")
    
    # Test needle generation
    needle_sample = generate_needle_sample(1024, needle_position=0.5, seed=42)
    assert "context" in needle_sample
    assert "query" in needle_sample
    assert "answer" in needle_sample
    print("✓ Needle generation works")


def test_codecs():
    """Test codec implementations"""
    print("Testing codecs...")
    
    test_text = "This is a test string for codec encoding and decoding."
    
    # Test TextCodec
    text_codec = TextCodec()
    encoded_text = text_codec.encode(test_text)
    assert encoded_text == test_text
    assert text_codec.decode(encoded_text) == test_text
    print("✓ TextCodec works")
    
    # Test RenderCodec
    render_codec = RenderCodec(font_size=14, line_width=40)
    encoded_img = render_codec.encode(test_text)
    assert encoded_img.size[0] > 0 and encoded_img.size[1] > 0
    print("✓ RenderCodec works")
    
    # Test CodebookCodec
    codebook_codec = CodebookCodec(color_mode="4color", cell_size=4, max_grid=64)
    encoded_codebook = codebook_codec.encode(test_text)
    assert encoded_codebook.size[0] > 0 and encoded_codebook.size[1] > 0
    print("✓ CodebookCodec works")
    
    # Test CodebookExternalCodec
    external_codec = CodebookExternalCodec(format="qrcode", max_bytes_per_symbol=100)
    encoded_external = external_codec.encode(test_text)
    if isinstance(encoded_external, list):
        assert len(encoded_external) > 0
    else:
        assert encoded_external.size[0] > 0
    print("✓ CodebookExternalCodec works")


def test_metrics():
    """Test evaluation metrics"""
    print("Testing metrics...")
    
    # Test JSON exact match
    json1 = '{"a": 1, "b": 2}'
    json2 = '{"b": 2, "a": 1}'
    assert json_exact_match(json1, json2) == 1.0
    print("✓ JSON exact match works")
    
    # Test JSON structural F1
    f1_result = json_structural_f1(json1, json2)
    assert "key_f1" in f1_result
    assert "overall_f1" in f1_result
    print("✓ JSON structural F1 works")
    
    # Test needle hit
    pred = "The secret code is SECRET-ABC123-XYZ789."
    gold = "SECRET-ABC123-XYZ789"
    assert needle_hit_at_1(pred, gold, fuzzy=True) == 1.0
    print("✓ Needle hit@1 works")


def main():
    """Run all smoke tests"""
    print("Running smoke tests...\n")
    
    try:
        test_data_generation()
        test_codecs()
        test_metrics()
        print("\n✓ All smoke tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
