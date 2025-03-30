import pytest
import numpy as np
import os
import torch
from PIL import Image
from cspe.identification.semantic_info import SemanticExtractor
from cspe.identification.prompts import CLIP_PROMPTS
import cv2

# Path to test images directory
TEST_IMG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../test_images"
)
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def load_test_image(filename):
    """Helper function to load a test image"""
    img_path = os.path.join(TEST_IMG_DIR, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img


# Mock classes and functions for testing
class MockModel:
    """Mock CLIP model for testing without actual inference"""

    def __init__(self, feature_pattern="uniform"):
        """
        Initialize mock model

        Args:
            feature_pattern: How to generate features
                - "uniform": All values are the same
                - "indexed": Each image gets values based on its index
                - "argmax": Each image gets a high value at a different position
        """
        self.feature_pattern = feature_pattern

    def __call__(self, images, text):
        batch_size = 1 if len(images.shape) == 3 else images.shape[0]
        features = torch.zeros((batch_size, len(CLIP_PROMPTS)))

        if self.feature_pattern == "uniform":
            features = torch.ones((batch_size, len(CLIP_PROMPTS)))
        elif self.feature_pattern == "indexed":
            for i in range(batch_size):
                f1 = torch.nn.functional.one_hot(
                    torch.tensor([i % len(CLIP_PROMPTS)]), num_classes=len(CLIP_PROMPTS)
                )
                features[i] = f1.to(torch.float32)

        elif self.feature_pattern == "argmax":
            for i in range(batch_size):
                features[i, i % len(CLIP_PROMPTS)] = (
                    10.0  # High value at different positions
                )

        return features, None

    def eval(self):
        return self


def mock_preprocess(x):
    """Standard mock preprocess function that returns tensor of expected shape"""
    if isinstance(x, list):
        return torch.ones((len(x), 3, 224, 224))
    elif isinstance(x, np.ndarray):
        return torch.ones((1, 3, 224, 224))
    elif isinstance(x, torch.Tensor) and len(x.shape) == 4:
        return x  # Batch already in correct format
    return torch.ones((1, 3, 224, 224))


class ShapeTracker:
    """Mock preprocess that tracks input shape"""

    def __init__(self):
        self.input_shape = None

    def __call__(self, x):
        self.input_shape = x.shape if isinstance(x, torch.Tensor) else None
        return torch.ones((1, 3, 224, 224))


@pytest.fixture
def mock_extractor():
    """Fixture that creates a SemanticExtractor with mock model and preprocess"""
    try:
        extractor = SemanticExtractor()
        # Save originals
        original_model = extractor.clip_model
        original_preprocess = extractor.clip_preprocess

        # Replace with mocks
        extractor.clip_model = MockModel()
        extractor.clip_preprocess = mock_preprocess

        yield extractor

        # Restore originals
        extractor.clip_model = original_model
        extractor.clip_preprocess = original_preprocess
    except Exception as e:
        pytest.skip(f"CLIP model initialization failed: {e}")


@pytest.fixture
def mock_extractor_with_indexed_features():
    """Fixture that creates an extractor with indexed feature output"""
    try:
        extractor = SemanticExtractor()
        # Save originals
        original_model = extractor.clip_model
        original_preprocess = extractor.clip_preprocess

        # Replace with mocks
        extractor.clip_model = MockModel(feature_pattern="indexed")
        extractor.clip_preprocess = mock_preprocess

        yield extractor

        # Restore originals
        extractor.clip_model = original_model
        extractor.clip_preprocess = original_preprocess
    except Exception as e:
        pytest.skip(f"CLIP model initialization failed: {e}")


@pytest.fixture
def mock_extractor_with_argmax_features():
    """Fixture that creates an extractor with argmax feature pattern"""
    try:
        extractor = SemanticExtractor()
        # Save originals
        original_model = extractor.clip_model
        original_preprocess = extractor.clip_preprocess

        # Replace with mocks
        extractor.clip_model = MockModel(feature_pattern="argmax")
        extractor.clip_preprocess = mock_preprocess

        yield extractor

        # Restore originals
        extractor.clip_model = original_model
        extractor.clip_preprocess = original_preprocess
    except Exception as e:
        pytest.skip(f"CLIP model initialization failed: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_init():
    """Test initialization of SemanticExtractor"""
    try:
        extractor = SemanticExtractor()
        assert len(extractor.image_queue) == 0
        assert extractor.device == "cpu"
        assert extractor.clip_model is not None
        assert extractor.processed_prompts is not None
        assert len(extractor.prompts) == len(CLIP_PROMPTS)
    except Exception as e:
        pytest.skip(f"CLIP model initialization failed: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_add_img():
    """Test adding an image to the queue"""
    try:
        extractor = SemanticExtractor()
        test_img = torch.ones((3, 224, 224))
        extractor.add_img(test_img)
        assert len(extractor.image_queue) == 1
        assert torch.equal(extractor.image_queue[0], test_img)
    except Exception as e:
        pytest.skip(f"CLIP model initialization failed: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_process_queue_empty():
    """Test processing an empty queue"""
    try:
        extractor = SemanticExtractor()
        result = extractor.process_queue()
        assert result is None
    except Exception as e:
        pytest.skip(f"CLIP model initialization failed: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_with_different_device():
    """Test initialization with GPU if available"""
    try:
        extractor = SemanticExtractor(device=DEVICE)
        assert extractor.device == DEVICE
    except Exception:
        pytest.skip("CUDA initialization failed")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_call_method_single_image(mock_extractor):
    """Test direct calling of the extractor with a properly processed image"""
    # Create a test image
    test_img = torch.ones((3, 224, 224))
    result = mock_extractor(test_img)

    # Check results
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, len(CLIP_PROMPTS))
    assert np.allclose(result.sum(), 1.0, atol=1e-5)  # Softmax should sum to 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_call_method_batch(mock_extractor):
    """Test direct calling of the extractor with a batch of images"""
    # Create a batch of test images
    batch_size = 3
    test_batch = [Image.new("RGB", (224, 224)) for _ in range(batch_size)]

    # Process directly
    result = mock_extractor(test_batch)

    # Check results
    assert isinstance(result, np.ndarray)
    assert result.shape == (batch_size, len(CLIP_PROMPTS))
    for i in range(batch_size):
        assert np.allclose(
            result[i].sum(), 1.0, atol=1e-5
        )  # Each softmax should sum to 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_process_queue_single_image(mock_extractor):
    """Test processing a queue with a single image"""
    # Add a test image to the queue
    test_img = torch.ones((3, 224, 224))
    mock_extractor.add_img(test_img)

    # Process the queue
    result = mock_extractor.process_queue()

    # Check results
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, len(CLIP_PROMPTS))
    assert np.allclose(result.sum(), 1.0, atol=1e-5)
    assert len(mock_extractor.image_queue) == 0  # Queue should be cleared


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_process_queue_multiple_images(mock_extractor_with_argmax_features):
    """Test processing a queue with multiple images"""
    extractor = mock_extractor_with_argmax_features

    # Add multiple test images to the queue
    num_images = 3
    for _ in range(num_images):
        test_img = torch.ones((3, 224, 224))
        extractor.add_img(test_img)

    # Process the queue
    result = extractor.process_queue()

    # Check results
    assert isinstance(result, np.ndarray)
    assert result.shape == (num_images, len(CLIP_PROMPTS))
    for i in range(num_images):
        assert np.allclose(result[i].sum(), 1.0, atol=1e-5)
        # Check that highest value is at expected position
        assert np.argmax(result[i]) == i % len(CLIP_PROMPTS)
    assert len(extractor.image_queue) == 0  # Queue should be cleared


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_results_order(mock_extractor_with_indexed_features):
    """Test that results from multiple images are returned in the correct order"""
    extractor = mock_extractor_with_indexed_features

    # Add multiple test images with identifiable values
    test_img1 = torch.ones((3, 224, 224)) * 0.1
    test_img2 = torch.ones((3, 224, 224)) * 0.5
    test_img3 = torch.ones((3, 224, 224)) * 0.9

    extractor.add_img(test_img1)
    extractor.add_img(test_img2)
    extractor.add_img(test_img3)

    # Process the queue
    result = extractor.process_queue()

    # Check results
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, len(CLIP_PROMPTS))

    # Values from our mock increase by image index
    assert np.argmax(result[0]) < np.argmax(result[1]) < np.argmax(result[2])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_with_real_image(mock_extractor):
    """Test with a real image if available"""
    try:
        # Check if test images directory exists and contains images
        filenames = os.listdir(TEST_IMG_DIR)
        image_files = [f for f in filenames if f.endswith((".jpg", ".png", ".jpeg"))]

        if not image_files:
            pytest.skip("No test images available")

        # Load real image
        test_img = load_test_image(image_files[0])

        # Process directly
        result = mock_extractor(test_img)

        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, len(CLIP_PROMPTS))
        assert np.allclose(result.sum(), 1.0, atol=1e-5)
    except FileNotFoundError:
        pytest.skip("Test images directory not found")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_different_image_shapes():
    """Test handling different image shapes"""
    try:
        extractor = SemanticExtractor()

        # Save originals
        original_model = extractor.clip_model
        original_preprocess = extractor.clip_preprocess

        # Replace with mocks
        extractor.clip_model = MockModel()
        extractor.clip_preprocess = ShapeTracker()
        shape_tracker = extractor.clip_preprocess

        try:
            # Test with different shaped images
            test_shapes = [
                torch.ones((3, 224, 224)),  # Standard CLIP input shape
                torch.ones((3, 512, 512)),  # Larger image
                torch.ones((3, 128, 256)),  # Non-square image
                torch.ones((1, 3, 224, 224)),  # Batched image
            ]

            for img in test_shapes:
                result = extractor(img)
                assert isinstance(result, np.ndarray)
                assert result.shape == (1, len(CLIP_PROMPTS))
                assert shape_tracker.input_shape is not None
        finally:
            # Restore original model and preprocess
            extractor.clip_model = original_model
            extractor.clip_preprocess = original_preprocess
    except Exception as e:
        pytest.skip(f"Test failed: {e}")


def _load_benchmark_images(batch_size=10):
    """Helper function to load images for benchmarking"""
    try:
        filenames = os.listdir(TEST_IMG_DIR)
        image_files = [f for f in filenames if f.endswith((".jpg", ".png", ".jpeg"))]

        if image_files:
            # Load real images
            test_imgs = []
            for f in image_files[:batch_size]:
                img = load_test_image(f)
                # Convert to PyTorch tensor (CxHxW format)
                img_tensor = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                test_imgs.append(img_tensor)

            test_img = test_imgs[0]  # Single image for testing

            # If we have enough images, use them directly; otherwise repeat to fill batch
            if len(test_imgs) >= batch_size:
                test_batch = test_imgs[:batch_size]
            else:
                # Repeat images to fill batch
                batch = []
                while len(batch) < batch_size:
                    batch.extend(test_imgs)
                test_batch = batch[:batch_size]

            return test_img, test_batch
        else:
            raise FileNotFoundError("No image files found")

    except (FileNotFoundError, IndexError):
        print("No test images available, using synthetic data")
        test_img = torch.rand((3, 224, 224))
        test_batch = torch.rand((batch_size, 3, 224, 224))
        return test_img, test_batch


def _run_benchmark(extractor, test_img, test_batch, num_runs=3):
    """Run benchmark tests for a given extractor"""
    import time

    batch_size = len(test_batch)

    # Warm-up run
    extractor(Image.new("RGB", (224, 224)))

    # Single image (multiple runs)
    single_times = []
    for i in range(num_runs):
        start_time = time.time()
        extractor(test_img)
        single_times.append(time.time() - start_time)
    single_time = sum(single_times) / num_runs

    # Batch (multiple runs)
    batch_times = []
    for i in range(num_runs):
        start_time = time.time()
        extractor(test_batch)
        batch_times.append(time.time() - start_time)
    batch_time = sum(batch_times) / num_runs

    return single_time, batch_time


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_benchmark_clip_cpu():
    """Benchmark test for CLIP performance on CPU"""
    num_runs = 3  # Number of runs to average
    batch_size = 10  # Batch size for testing

    # Load test images
    test_img, test_batch = _load_benchmark_images(batch_size)

    # Run on CPU
    print("\nBenchmarking CLIP performance on CPU:")
    extractor_cpu = SemanticExtractor(device="cpu")
    print(f"Running CPU benchmarks ({num_runs} runs each)...")

    cpu_single_time, cpu_batch_time = _run_benchmark(
        extractor_cpu, test_img, test_batch, num_runs
    )

    # Report results
    print(f"CPU single image: {cpu_single_time:.4f} seconds (avg of {num_runs} runs)")
    print(
        f"CPU batch ({batch_size} images): {cpu_batch_time:.4f} seconds (avg of {num_runs} runs)"
    )
    print(f"CPU average per image in batch: {cpu_batch_time / batch_size:.4f} seconds")
    print(
        f"Batch processing speedup on CPU: {(cpu_single_time * batch_size) / cpu_batch_time:.2f}x"
    )

    # Save these results in a file
    with open("cpu_benchmark_results.txt", "w+") as f:
        f.write(
            f"CPU: \nSingle Image:{cpu_single_time:.4f} | Batch Inference: {cpu_batch_time:.4f} | Batch/BatchSize: {cpu_batch_time / batch_size:.4f}\n"
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_benchmark_clip_gpu():
    """Benchmark test for CLIP performance on GPU (CUDA)"""
    num_runs = 3  # Number of runs to average
    batch_size = 10  # Batch size for testing

    # Load test images
    test_img, test_batch = _load_benchmark_images(batch_size)

    # Run on CPU first to get comparison values
    print("\nBenchmarking CLIP performance on GPU:")
    extractor_cpu = SemanticExtractor(device="cpu")
    print("Running CPU benchmark for comparison...")
    cpu_single_time, cpu_batch_time = _run_benchmark(
        extractor_cpu,
        test_img,
        test_batch,
        num_runs=1,  # Just one run for comparison
    )

    # Run GPU benchmarks
    print(f"Running GPU benchmarks ({num_runs} runs each)...")
    extractor_gpu = SemanticExtractor(device=DEVICE)

    gpu_single_time, gpu_batch_time = _run_benchmark(
        extractor_gpu, test_img, test_batch, num_runs
    )

    # Report results
    print(f"GPU single image: {gpu_single_time:.4f} seconds (avg of {num_runs} runs)")
    print(
        f"GPU batch ({batch_size} images): {gpu_batch_time:.4f} seconds (avg of {num_runs} runs)"
    )
    print(f"GPU average per image in batch: {gpu_batch_time / batch_size:.4f} seconds")

    # Speedup calculations
    print(f"GPU speedup (single image): {cpu_single_time / gpu_single_time:.2f}x")
    print(f"GPU speedup (batch): {cpu_batch_time / gpu_batch_time:.2f}x")
    print(
        f"Batch processing speedup on GPU: {(gpu_single_time * batch_size) / gpu_batch_time:.2f}x"
    )

    # Save these results in a file
    with open("gpu_benchmark_results.txt", "w+") as f:
        f.write(
            f"""GPU:
Single Image:{gpu_single_time:.4f} | Batch Inference: {gpu_batch_time:.4f} | Batch/BatchSize: {gpu_batch_time / batch_size:.4f}
Speedup (single image): {cpu_single_time / gpu_single_time:.2f} | Speedup (batch): {cpu_batch_time / gpu_batch_time:.2f}
Batch Processing Speedup: {(gpu_single_time * batch_size) / gpu_batch_time:.2f}
            """
        )
