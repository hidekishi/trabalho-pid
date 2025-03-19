from canny import canny_edge_detector
from marr_hildreth import marr_hildreth
from otsu import otsu_thresholding
from watershed import watershed_segmentation
import cv2
import numpy as np

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
    combined_image = np.hstack((image1, image2))
    
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

show_two_images(marr_hildreth_image, canny_image, "Marr-Hildreth / Canny")