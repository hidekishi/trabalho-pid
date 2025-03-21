import numpy as np
from scipy.ndimage import label, distance_transform_edt
from collections import deque
import cv2

def watershed(image):
    # Passo 1: Calcular o gradiente da imagem (pode ser substituído por outro método de detecção de bordas)
    gradient = np.gradient(image)
    gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
    
    # Passo 2: Identificar os marcadores (mínimos regionais)
    markers = np.zeros_like(image, dtype=int)
    markers[image == 0] = 1  # Marcadores para o fundo
    markers[image == 255] = 2  # Marcadores para os objetos
    
    # Passo 3: Inicializar a fila de pixels a serem processados
    queue = deque()
    
    # Passo 4: Inundação a partir dos marcadores
    for marker in np.unique(markers):
        if marker == 0:
            continue
        mask = (markers == marker)
        distance = distance_transform_edt(mask)
        local_maxima = (distance == np.max(distance))
        queue.extend(zip(*np.where(local_maxima)))
    
    # Passo 5: Processar a fila
    while queue:
        x, y = queue.popleft()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= image.shape[0] or ny >= image.shape[1]:
                    continue
                if markers[nx, ny] == 0:
                    markers[nx, ny] = markers[x, y]
                    queue.append((nx, ny))
    
    return markers

# Exemplo de uso
if __name__ == "__main__":
    # Criar uma imagem binária de exemplo
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:40, 20:40] = 255
    image[60:80, 60:80] = 255
    
    # Aplicar o algoritmo Watershed
    segmented_image = watershed(image)
    
    # Normalizar a imagem segmentada para exibição
    segmented_image_display = cv2.normalize(segmented_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Exibir a imagem original e a segmentada usando OpenCV
    cv2.imshow("Imagem Original", image)
    cv2.imshow("Imagem Segmentada (Watershed)", segmented_image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()