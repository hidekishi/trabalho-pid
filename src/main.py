from algoritmos import canny_edge_detector, marr_hildreth, otsu_thresholding, watershed_segmentation, contar_objetos, segmentar_imagem, freeman_chain_code
import cv2
import numpy as np

def is_binary(image):
    """
    Verifica se a imagem é binária.
    
    Parâmetros:
        image: Imagem (numpy array).
    
    Retorna:
        bool: True se a imagem for binária, False caso contrário.
    """
    unique_values = np.unique(image)
    return set(unique_values) <= {0, 1} or set(unique_values) <= {0, 255}

def show_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_two_images(image1, image2, title):
    # Verifica se as imagens têm a mesma altura
    if image1.shape[0] != image2.shape[0]:
        # Redimensiona a segunda imagem para ter a mesma altura da primeira
        scale_factor = image1.shape[0] / image2.shape[0]
        new_width = int(image2.shape[1] * scale_factor)
        image2 = cv2.resize(image2, (new_width, image1.shape[0]))
    
    # Combina as duas imagens horizontalmente
    combined_image = cv2.hconcat([image1, image2])
    
    # Exibe a imagem combinada
    cv2.imshow(title, combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

filename = "../imagens/"+input("Nome do arquivo de imagem original: ")
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

otsu_image = otsu_thresholding(image)
watershed_image = watershed_segmentation(image)
marr_hildreth_image = marr_hildreth(image)
canny_image = canny_edge_detector(image, 50, 150)

show_image(otsu_image, "Otsu")
show_image(watershed_image, "Watershed")
show_image(watershed_segmentation(otsu_image), "Watershed Binario")

show_two_images(marr_hildreth_image, canny_image, "Marr-Hildreth / Canny")

show_two_images(image, segmentar_imagem(image), "Greyscale / Greyscale Segmentada")

label_num = contar_objetos(otsu_image)
print(label_num)