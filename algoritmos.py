import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel, label, distance_transform_edt

def read_image(image_path):
    from PIL import Image
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=np.float64)

def marr_hildreth(image):
    blurred = gaussian_filter(image, sigma=1.0)
    laplacian = sobel(blurred, axis=0) + sobel(blurred, axis=1)
    edges = np.uint8(np.abs(laplacian))
    return edges

def canny(image, low_threshold=50, high_threshold=150):
    Gx = sobel(image, axis=0)
    Gy = sobel(image, axis=1)
    gradient_magnitude = np.hypot(Gx, Gy)
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    edges = (gradient_magnitude > low_threshold) & (gradient_magnitude < high_threshold)
    return np.uint8(edges * 255)

def otsu_thresholding(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    total = image.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground, weight_background = 0, 0, 0
    for i in range(256):
        sum_total += i * hist[i]
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_foreground += i * hist[i]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance_between > current_max:
            current_max = variance_between
            threshold = i
    return (image > threshold) * 255

def watershed_segmentation(image):
    threshold = otsu_thresholding(image)
    markers, _ = label(threshold)
    distance = distance_transform_edt(threshold)
    labels, _ = label(distance > (0.7 * distance.max()))
    segmented = np.zeros_like(image)
    for label_val in range(1, labels.max() + 1):
        segmented[labels == label_val] = 255 * label_val / labels.max()
    return segmented

def display_results(image_path):
    image = read_image(image_path)
    methods = {
        "Marr-Hildreth": marr_hildreth(image),
        "Canny": canny(image),
        "Otsu": otsu_thresholding(image),
        "Watershed": watershed_segmentation(image)
    }
    plt.figure(figsize=(10, 8))
    for i, (title, result) in enumerate(methods.items(), 1):
        plt.subplot(2, 2, i)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis("off")
    plt.show()

# Exemplo de uso:
# display_results("caminho_para_imagem.jpg")
