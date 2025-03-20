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

def canny_edge_detector(image, low_threshold, high_threshold):
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

def contar_objetos(imagem_binaria):
    # Dimensões da imagem
    altura, largura = imagem_binaria.shape
    
    # Matriz para marcar pixels já visitados
    visitado = np.zeros_like(imagem_binaria, dtype=bool)
    
    # Direções para movimentação (8 conectividade)
    direcoes = [(-1, -1), (-1, 0), (-1, 1), 
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)]
    
    def dfs(x, y):
        """Explora a região conectada a partir do pixel (x, y) usando DFS."""
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in direcoes:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < altura and 0 <= ny < largura and not visitado[nx, ny] and imagem_binaria[nx, ny] == 255:
                    visitado[nx, ny] = True
                    stack.append((nx, ny))
    
    # Contador de objetos
    num_objetos = 0
    
    # Percorre a matriz procurando objetos
    for i in range(altura):
        for j in range(largura):
            if imagem_binaria[i, j] == 255 and not visitado[i, j]:  # Novo objeto encontrado
                visitado[i, j] = True
                num_objetos += 1
                dfs(i, j)  # Preenche toda a região conectada
    
    return num_objetos

def freeman_chain_code(imagem_binaria):
    def encontrar_ponto_inicial(imagem_binaria):
        for i in range(imagem_binaria.shape[0]):
            for j in range(imagem_binaria.shape[1]):
                if imagem_binaria[i, j] == 255:
                    return (i, j)
        return None

    DIRECTIONS = [
    (0, 1),   # 0: Leste
    (-1, 1),  # 1: Nordeste
    (-1, 0),  # 2: Norte
    (-1, -1), # 3: Noroeste
    (0, -1),  # 4: Oeste
    (1, -1),  # 5: Sudoeste
    (1, 0),   # 6: Sul
    (1, 1)    # 7: Sudeste
    ]
    """
    Calcula a cadeia de Freeman para o contorno de um objeto em uma imagem binária.
    """
    # Encontra o ponto inicial
    ponto_inicial = encontrar_ponto_inicial(imagem_binaria)
    if ponto_inicial is None:
        return []  # Nenhum objeto encontrado
    
    cadeia = []
    ponto_atual = ponto_inicial
    direcao_inicial = 0  # Começa na direção 0 (Leste)
    
    while True:
        # Encontra a próxima direção
        encontrou_proximo = False
        for d in range(8):
            # Calcula a próxima direção (usando módulo para circularidade)
            direcao = (direcao_inicial + d) % 8
            dy, dx = DIRECTIONS[direcao]
            y, x = ponto_atual[0] + dy, ponto_atual[1] + dx
            
            # Verifica se o próximo pixel está dentro da imagem e é parte do objeto
            if 0 <= y < imagem_binaria.shape[0] and 0 <= x < imagem_binaria.shape[1]:
                if imagem_binaria[y, x] == 255:
                    cadeia.append(direcao)
                    ponto_atual = (y, x)
                    direcao_inicial = (direcao + 5) % 8  # Ajusta a direção inicial
                    encontrou_proximo = True
                    break
        
        # Se não encontrou próximo pixel, termina
        if not encontrou_proximo:
            break
        
        # Se retornou ao ponto inicial, termina
        if ponto_atual == ponto_inicial:
            break
    
    return cadeia

def segmentar_imagem(imagem):
    # Cria uma cópia da imagem para não modificar a original
    imagem_segmentada = imagem.copy()
    
    # Define as faixas de intensidade e os valores correspondentes
    faixas = [
        (0, 50, 25),
        (51, 100, 75),
        (101, 150, 125),
        (151, 200, 175),
        (201, 255, 255)
    ]
    
    # Percorre cada pixel da imagem
    for i in range(imagem.shape[0]):  # Linhas
        for j in range(imagem.shape[1]):  # Colunas
            intensidade = imagem[i, j]
            
            # Verifica em qual faixa a intensidade do pixel se encaixa
            for inicio, fim, valor in faixas:
                if inicio <= intensidade <= fim:
                    imagem_segmentada[i, j] = valor
                    break
    
    return imagem_segmentada