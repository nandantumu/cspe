import pytest
import numpy as np
import os
from PIL import Image
from cspe.identification.low_level_info import LowLevelExtractor
from cspe.identification.prompts import CLIP_PROMPTS
import cv2

# Path to test images directory
TEST_IMG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../test_images"
)


def load_test_image(filename):
    """Helper function to load a test image"""
    img_path = os.path.join(TEST_IMG_DIR, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img


def test_init():
    """Test initialization of LowLevelExtractor"""
    extractor = LowLevelExtractor()
    assert len(extractor.image_queue) == 0


def test_add_img():
    """Test adding an image to the queue"""
    extractor = LowLevelExtractor()
    test_img = np.ones((10, 20, 3))
    extractor.add_img(test_img)
    assert len(extractor.image_queue) == 1
    assert np.array_equal(extractor.image_queue[0], test_img)


def test_process_queue_empty():
    """Test processing an empty queue"""
    extractor = LowLevelExtractor()
    result = extractor.process_queue()
    assert result is None


def test_process_queue_single_image():
    """Test processing a queue with a single image"""
    extractor = LowLevelExtractor()
    test_img = np.ones((3, 4, 3))
    test_img[0, 0, 0] = 0
    test_img[1, 1, 1] = 0
    test_img[2, 2, 2] = 0

    extractor.add_img(test_img)
    result = extractor.process_queue()

    # Expected average: (3*4-1) ones and 1 zero for each channel = 11/12
    expected_avg = np.array([[11 / 12, 11 / 12, 11 / 12]])

    assert np.allclose(result, expected_avg)
    assert len(extractor.image_queue) == 0


def test_process_queue_multiple_images():
    """Test processing a queue with multiple images"""
    extractor = LowLevelExtractor()

    test_img1 = np.ones((3, 4, 3))
    test_img2 = np.zeros((3, 4, 3))

    extractor.add_img(test_img1)
    extractor.add_img(test_img2)
    result = extractor.process_queue()

    expected_avg = np.array(
        [
            [1.0, 1.0, 1.0],  # Average RGB for test_img1
            [0.0, 0.0, 0.0],  # Average RGB for test_img2
        ]
    )

    assert np.allclose(result, expected_avg)
    assert len(extractor.image_queue) == 0


def test_call_method_batch():
    """Test direct calling of the extractor"""
    extractor = LowLevelExtractor()

    test_img = np.ones((1, 3, 4, 3))  # Add batch dimension
    test_img[0, 0, 0, 0] = 0

    result = extractor(test_img)
    expected_avg = test_img[0].mean(axis=(0, 1))

    assert np.allclose(result, expected_avg)


def test_call_method_single_image():
    """Test direct calling of the extractor with a single image"""
    extractor = LowLevelExtractor()

    test_img = np.ones((3, 4, 3))
    test_img[0, 0, 0] = 0

    result = extractor(test_img)
    expected_avg = np.array([[11 / 12, 1.0, 1.0]])

    assert np.allclose(result, expected_avg)


def test_with_real_images():
    """Test with a real image from test directory"""
    try:
        filenames = os.listdir(TEST_IMG_DIR)
        image_files = [f for f in filenames if f.endswith((".jpg", ".png", ".jpeg"))]

        if not image_files:
            pytest.skip("No image files found in test directory")

        test_imgs = [load_test_image(f) for f in image_files]
        extractor = LowLevelExtractor()

        # Test queue method
        test_img_batch = np.stack(test_imgs)
        for img in test_imgs:
            extractor.add_img(img)
        result = extractor.process_queue()
        expected_avg = test_img_batch.mean(axis=(1, 2))
        assert np.allclose(result, expected_avg)

    except FileNotFoundError:
        pytest.skip("Test images directory not found")


def test_real_image():
    """ """
    try:
        filenames = os.listdir(TEST_IMG_DIR)
        image_files = [f for f in filenames if f.endswith((".jpg", ".png", ".jpeg"))]

        if not image_files:
            pytest.skip("No image files found in test directory")

        extractor = LowLevelExtractor()

        # Test single image
        test_img = load_test_image(image_files[0])
        result = extractor(test_img)
        expected_avg = np.array([test_img.mean(axis=(0, 1))])
        assert np.allclose(result, expected_avg)
    except FileNotFoundError:
        pytest.skip("Test images directory not found")


def test_results_order():
    """Test that results from multiple images are returned in the correct order"""
    extractor = LowLevelExtractor()

    # Create three distinct test images
    img1 = np.ones((3, 4, 3)) * 0.1  # All values 0.1
    img2 = np.ones((3, 4, 3)) * 0.5  # All values 0.5
    img3 = np.ones((3, 4, 3)) * 0.9  # All values 0.9

    # Add images in specific order
    extractor.add_img(img1)
    extractor.add_img(img2)
    extractor.add_img(img3)

    result = extractor.process_queue()

    # Expected averages in the same order as added
    expected_avg = np.array(
        [
            [0.1, 0.1, 0.1],  # Average RGB for img1
            [0.5, 0.5, 0.5],  # Average RGB for img2
            [0.9, 0.9, 0.9],  # Average RGB for img3
        ]
    )

    assert np.allclose(result, expected_avg)
    assert len(extractor.image_queue) == 0


def test_extreme_values():
    """Test processing images with extreme values"""
    extractor = LowLevelExtractor()

    # Create image with extreme values
    img = np.zeros((5, 5, 3))
    img[0, 0, :] = 1.0  # Max value in one pixel

    extractor.add_img(img)
    result = extractor.process_queue()

    # Expected: 1 pixel is 1.0, 24 pixels are 0, so average is 1/25
    expected_avg = np.array([[1 / 25, 1 / 25, 1 / 25]])

    assert np.allclose(result, expected_avg)
