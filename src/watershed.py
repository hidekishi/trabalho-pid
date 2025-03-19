import numpy as np
import cv2

def watershed_segmentation(image):
    """Implementação do algoritmo Watershed para segmentação."""
    # Binarização da imagem
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remoção de ruído
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Área de fundo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Área de primeiro plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Área desconhecida
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # Aplicação do Watershed
    # Converta a imagem para colorida (3 canais) antes de aplicar o Watershed
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_color, markers)
    
    # Marca as bordas dos objetos na imagem colorida
    image_color[markers == -1] = [0, 0, 255]  # Bordas em vermelho
    
    return image_color