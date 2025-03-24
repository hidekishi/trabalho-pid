import cv2
import numpy as np

def erode(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Criar imagem de saída preenchida com zeros
    output = np.zeros((h, w), dtype=np.uint8)
    
    # Preencher bordas refletindo os pixels da imagem original
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Percorrer a imagem incluindo as bordas
    for i in range(h):
        for j in range(w):
            # Extrair a região de interesse (ROI)
            roi = padded_image[i:i + kh, j:j + kw]
            # Aplicar erosão: verifica se todos os pixels sob o kernel são brancos
            if np.any(roi[kernel == 1] == 255):
                output[i, j] = 255
    
    return output

def dilate(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Criar imagem de saída preenchida com zeros
    output = np.zeros((h, w), dtype=np.uint8)
    
    # Preencher bordas refletindo os pixels da imagem original
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Percorrer a imagem incluindo as bordas
    for i in range(h):
        for j in range(w):
            # Extrair a região de interesse (ROI)
            roi = padded_image[i:i + kh, j:j + kw]
            # Aplicar dilatação: verifica se pelo menos um pixel sob o kernel é branco
            if np.all(roi[kernel == 1] == 255):
                output[i, j] = 255
    
    return output

def otsu(image):
    # Histograma da imagem
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / np.sum(hist)

    best_threshold = 0
    max_variance = 0
    # Testa diferentes limiares
    for threshold in range(256):
        w0 = np.sum(hist[:threshold])
        w1 = np.sum(hist[threshold:])
        if w0 == 0 or w1 == 0:
            continue
        # Calcula a variancia
        mean0 = np.sum(np.arange(threshold) * hist[:threshold]) / w0
        mean1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1
        variance = w0 * w1 * (mean0 - mean1)**2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    # Aplica o limiar
    binary_image = np.zeros_like(image)
    binary_image[image >= best_threshold] = 255
    
    return binary_image

def marr_hildreth(image):
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    def laplacian_of_gaussian(image, size=5, sigma=1):
        kernel = gaussian_kernel(size, sigma)
        kernel = cv2.Laplacian(kernel, cv2.CV_64F)
        log_image = cv2.filter2D(image, -1, kernel)
        return log_image
    
    log_image = laplacian_of_gaussian(image)
    return log_image

def canny_edge_detector(image, low_threshold, high_threshold):
    def non_maximum_suppression(magnitude, direction):
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

def contar_objetos(image):
    altura, largura = image.shape
    visitado = np.zeros_like(image, dtype=bool)
    
    # Vizinhanca
    direcoes = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Busca em profundidade para encontrar todos os pixels conectados
    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in direcoes:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < altura and 0 <= ny < largura and not visitado[nx, ny] and image[nx, ny] == 0:
                    visitado[nx, ny] = True
                    stack.append((nx, ny))
    
    num_objetos = 0
    
    # Percorre a matriz procurando objetos
    for i in range(altura):
        for j in range(largura):
            if image[i, j] == 0 and not visitado[i, j]:
                visitado[i, j] = True
                num_objetos += 1
                dfs(i, j)
    return num_objetos

def segmentar_imagem(imagem):
    imagem_segmentada = imagem.copy()
    # Faixas de intensidade (Min, Max, Valor)
    faixas = [(0, 50, 25),
              (51, 100, 75),
              (101, 150, 125),
              (151, 200, 175),
              (201, 255, 255)
    ]
    # Substitui intensidades
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            intensidade = imagem[i, j]
            for inicio, fim, valor in faixas:
                if inicio <= intensidade <= fim:
                    imagem_segmentada[i, j] = valor
                    break
    return imagem_segmentada

def filtro_box(imagem, kernel_size):
    # Obtém as dimensões da imagem
    altura, largura = imagem.shape

    # Cria uma imagem de saída preenchida com zeros
    imagem_filtrada = np.zeros_like(imagem, dtype=np.float32)

    # Define o raio do kernel
    raio = kernel_size // 2

    # Percorre a imagem, aplicando o filtro box
    for y in range(raio, altura - raio):
        for x in range(raio, largura - raio):
            # Extrai a região de interesse (ROI) sob o kernel
            roi = imagem[y - raio:y + raio + 1, x - raio:x + raio + 1]

            # Calcula a média dos valores na ROI
            media = np.mean(roi)

            # Atribui o valor médio ao pixel central na imagem filtrada
            imagem_filtrada[y, x] = media

    # Converte a imagem filtrada de volta para uint8
    imagem_filtrada = np.uint8(imagem_filtrada)

    return imagem_filtrada