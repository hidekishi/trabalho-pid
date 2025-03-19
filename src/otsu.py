import cv2
import numpy as np

def otsu_thresholding(image):
    """Implementação do método de Otsu para binarização."""
    # Histograma da imagem
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    # Normalização do histograma
    hist = hist.astype(np.float32) / np.sum(hist)
    
    # Cálculo da variância intra-classe
    best_threshold = 0
    max_variance = 0
    
    for threshold in range(256):
        w0 = np.sum(hist[:threshold])
        w1 = np.sum(hist[threshold:])
        
        if w0 == 0 or w1 == 0:
            continue
        
        mean0 = np.sum(np.arange(threshold) * hist[:threshold]) / w0
        mean1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1
        
        variance = w0 * w1 * (mean0 - mean1)**2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    
    # Aplicação do limiar
    binary_image = np.zeros_like(image)
    binary_image[image >= best_threshold] = 255
    
    return binary_image
