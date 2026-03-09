import numpy as np
from blitz.data.tools import resize_and_convert_to_8_bit, resize_and_convert

def test_resize_and_convert_to_8_bit_downsample():
    # Create a 100x100 array
    array = np.random.rand(100, 100).astype(np.float32)
    size_ratio = 0.5
    resized = resize_and_convert_to_8_bit(array, size_ratio, convert_to_8_bit=False)
    assert resized.shape == (50, 50)
    assert resized.dtype == np.float32

def test_resize_and_convert_to_8_bit_upsample():
    # Create a 10x10 array
    array = np.random.rand(10, 10).astype(np.float32)
    size_ratio = 2.0
    resized = resize_and_convert_to_8_bit(array, size_ratio, convert_to_8_bit=False)
    assert resized.shape == (20, 20)
    assert resized.dtype == np.float32

def test_resize_and_convert_to_8_bit_8bit_conversion():
    # Create a float32 array with range [0, 1]
    array = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
    # size_ratio = 1.0 (no resizing, just conversion)
    resized = resize_and_convert_to_8_bit(array, 1.0, convert_to_8_bit=True)
    assert resized.dtype == np.uint8
    assert resized.shape == (2, 2)
    # 1.0 should become 255, 0.0 should stay 0
    assert resized[1, 0] == 255
    assert resized[0, 0] == 0
    # 0.5 should become approx 127 or 128
    assert 127 <= resized[0, 1] <= 128

def test_resize_and_convert_to_8_bit_color_image():
    # Create a 100x100x3 color image
    array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    size_ratio = 0.1
    resized = resize_and_convert_to_8_bit(array, size_ratio, convert_to_8_bit=False)
    assert resized.shape == (10, 10, 3)
    assert resized.dtype == np.uint8

def test_resize_and_convert_to_8_bit_all_zeros():
    # All zeros should not crash and should return all zeros
    array = np.zeros((10, 10), dtype=np.float32)
    resized = resize_and_convert_to_8_bit(array, 1.0, convert_to_8_bit=True)
    assert np.all(resized == 0)
    assert resized.dtype == np.uint8

def test_resize_and_convert_basic():
    array = np.ones((100, 100), dtype=np.float32)
    resized = resize_and_convert(array, 0.5, False)
    assert resized.shape == (50, 50)

def test_resize_and_convert_8bit():
    array = np.zeros((100, 100), dtype=np.float32)
    array[0, 0] = 1.0
    resized = resize_and_convert(array, 1.0, True)
    assert resized.dtype == np.uint8
    assert np.max(resized) == 255

def test_resize_and_convert_comparison():
    # Compare resize_and_convert and resize_and_convert_to_8_bit
    array = np.random.rand(100, 100).astype(np.float32)

    # For ratio 1.0 where interpolation doesn't matter, they should be identical
    res1_id = resize_and_convert(array, 1.0, convert_to_8_bit=True)
    res2_id = resize_and_convert_to_8_bit(array, 1.0, convert_to_8_bit=True)
    np.testing.assert_array_equal(res1_id, res2_id)
