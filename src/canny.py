import cv2
import numpy as np

def canny_edge_detector(image, low_threshold, high_threshold):
    """Implementação simplificada do detector de bordas Canny."""
    # Suavização com filtro Gaussiano
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Cálculo do gradiente usando Sobel
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude e direção do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Supressão não máxima
    suppressed = non_maximum_suppression(magnitude, direction)
    
    # Limiarização com histerese
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    
    return edges

def non_maximum_suppression(magnitude, direction):
    """Supressão não máxima."""
    M, N = magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # Ângulo 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Ângulo 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Ângulo 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Ângulo 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                suppressed[i,j] = magnitude[i,j]
            else:
                suppressed[i,j] = 0

    return suppressed

def hysteresis_thresholding(image, low, high):
    """Limiarização com histerese."""
    M, N = image.shape
    strong = np.zeros((M, N), dtype=np.uint8)
    weak = np.zeros((M, N), dtype=np.uint8)

    strong[image >= high] = 255
    weak[(image >= low) & (image < high)] = 255

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (weak[i,j] == 255):
                if ((strong[i-1,j-1] == 255) or (strong[i-1,j] == 255) or (strong[i-1,j+1] == 255) or
                    (strong[i,j-1] == 255) or (strong[i,j+1] == 255) or
                    (strong[i+1,j-1] == 255) or (strong[i+1,j] == 255) or (strong[i+1,j+1] == 255)):
                    strong[i,j] = 255
                else:
                    strong[i,j] = 0

    return strong