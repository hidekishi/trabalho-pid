import numpy as np
import cv2

def marr_hildreth(image):
    def gaussian_kernel(size, sigma):
        """Cria um kernel Gaussiano."""
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    def laplacian_of_gaussian(image, size=5, sigma=1):
        """Aplica o filtro Laplaciano de Gaussiano (LoG) na imagem."""
        kernel = gaussian_kernel(size, sigma)
        kernel = cv2.Laplacian(kernel, cv2.CV_64F)
        log_image = cv2.filter2D(image, -1, kernel)
        return log_image
    
    log_image = laplacian_of_gaussian(image)
    return log_image