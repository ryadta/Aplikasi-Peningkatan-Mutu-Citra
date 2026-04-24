import cv2
import numpy as np


class ImageProcessingError(Exception):
    """Raised when image processing cannot be performed."""


def ensure_odd(value: int, minimum: int = 1) -> int:
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    return value


def apply_smoothing(image: np.ndarray, kernel_size: int) -> np.ndarray:
    ksize = ensure_odd(kernel_size, minimum=1)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_sharpening(image: np.ndarray, strength: int) -> np.ndarray:
    alpha = max(strength, 0) / 10.0
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_sobel(image: np.ndarray, kernel_size: int) -> np.ndarray:
    ksize = ensure_odd(kernel_size, minimum=3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    edge = magnitude.astype(np.uint8)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)


def apply_laplacian(image: np.ndarray, kernel_size: int) -> np.ndarray:
    ksize = ensure_odd(kernel_size, minimum=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap = np.absolute(lap)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    edge = lap.astype(np.uint8)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)


def _build_lookup(source_channel: np.ndarray, reference_channel: np.ndarray) -> np.ndarray:
    source_hist, _ = np.histogram(source_channel.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])

    source_cdf = np.cumsum(source_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)

    source_cdf /= source_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    lookup = np.zeros(256, dtype=np.uint8)
    for source_intensity in range(256):
        closest = np.argmin(np.abs(source_cdf[source_intensity] - ref_cdf))
        lookup[source_intensity] = closest
    return lookup


def apply_histogram_specification(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image is None or reference is None:
        raise ImageProcessingError("Input image and reference image are required.")

    if image.ndim != 3 or reference.ndim != 3:
        raise ImageProcessingError("Histogram specification expects color images.")

    result = np.zeros_like(image)
    channels = min(image.shape[2], reference.shape[2], 3)

    for channel in range(channels):
        lookup = _build_lookup(image[:, :, channel], reference[:, :, channel])
        result[:, :, channel] = cv2.LUT(image[:, :, channel], lookup)

    return result


def apply_convolution(image: np.ndarray, filter_name: str, kernel_size: int, strength: int) -> np.ndarray:
    if filter_name == "Smoothing":
        return apply_smoothing(image, kernel_size)
    if filter_name == "Sharpening":
        return apply_sharpening(image, strength)
    if filter_name == "Sobel Edge":
        return apply_sobel(image, kernel_size)
    if filter_name == "Laplacian Edge":
        return apply_laplacian(image, kernel_size)
    return image.copy()


def create_histogram_preview(
    image: np.ndarray,
    mode: str = "RGB",
    width: int = 460,
    height: int = 220,
) -> np.ndarray:
    if image is None:
        raise ImageProcessingError("Image is required for histogram preview.")

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    plot_height = height - 24
    bins = np.arange(256)
    if mode == "Grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        if hist.max() > 0:
            hist = hist / hist.max()

        points = []
        for x in bins:
            px = int((x / 255.0) * (width - 1))
            py = int(plot_height - (hist[x] * (plot_height - 1)))
            points.append([px, py])

        line_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [line_points], isClosed=False, color=(230, 230, 230), thickness=1)
    else:
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # B, G, R
        for channel, color in enumerate(colors):
            hist = cv2.calcHist([image], [channel], None, [256], [0, 256]).flatten()
            if hist.max() > 0:
                hist = hist / hist.max()

            points = []
            for x in bins:
                px = int((x / 255.0) * (width - 1))
                py = int(plot_height - (hist[x] * (plot_height - 1)))
                points.append([px, py])

            line_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [line_points], isClosed=False, color=color, thickness=1)

    cv2.line(canvas, (0, plot_height), (width - 1, plot_height), (180, 180, 180), 1)
    return canvas
