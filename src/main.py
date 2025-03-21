from algoritmos import filtro_box, erode, dilate, canny_edge_detector, marr_hildreth, otsu, watershed_segmentation, contar_objetos, segmentar_imagem, freeman_chain_code
import cv2
import numpy as np

def is_binary(image):
    unique_values = np.unique(image)
    return set(unique_values) <= {0, 1} or set(unique_values) <= {0, 255}

def save_image(image, title):
    cv2.imwrite("../saidas/"+title+".jpg", image)

def show_image(image, title):
    cv2.imshow(title, image)
    save_image(image, title)
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
    save_image(image1, title+"1")
    save_image(image2, title+"2")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_four_images(image1, image2, image3, image4, title):
    def merge_image(image1, image2):
        if image1.shape[0] != image2.shape[0]:
            # Redimensiona a segunda imagem para ter a mesma altura da primeira
            scale_factor = image1.shape[0] / image2.shape[0]
            new_width = int(image2.shape[1] * scale_factor)
            image2 = cv2.resize(image2, (new_width, image1.shape[0]))
        
        # Combina as duas imagens horizontalmente
        return cv2.hconcat([image1, image2])
    save_image(image1, title+"2")
    save_image(image2, title+"3")
    save_image(image1, title+"5")
    save_image(image2, title+"7")
    combined_image = merge_image(merge_image(image1, image2), merge_image(image3, image4))
    
    # Exibe a imagem combinada
    cv2.imshow(title, combined_image)
    save_image(image1, title+"1")
    save_image(image2, title+"2")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

filename = "../imagens/"+input("Nome do arquivo de imagem original: ")
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]], dtype=np.uint8)

otsu_image = otsu(image)
print("Binária com Otsu calculada...")
otsu_eroded_image = erode(otsu_image, kernel)
print("Binária erodida...")
otsu_eroded_dilated_image = dilate(otsu_eroded_image, kernel)
print("Binária erodida e dilatada...")
watershed_image = watershed_segmentation(image)
print("Watershed executado...")
marr_hildreth_image = marr_hildreth(image)
print("Marr-Hildreth executado...")
canny_image = canny_edge_detector(image, 50, 150)
print("Canny executado...")
segmented_image = segmentar_imagem(image)
print("Segmentação em binária executada...")
box_2 = filtro_box(image, 2)
box_3 = filtro_box(image, 3)
box_5 = filtro_box(image, 5)
box_7 = filtro_box(image, 7)

show_image(otsu_image, "Otsu")
show_image(otsu_eroded_image, "Otsu erodido")
show_image(watershed_image, "Watershed")

show_two_images(otsu_image, otsu_eroded_dilated_image, "Otsu normal-Aberto")
show_two_images(marr_hildreth_image, canny_image, "Marr-Hildreth-Canny")
show_two_images(image, segmented_image, "Greyscale-Greyscale Segmentada")

show_four_images(box_2, box_3, box_5, box_7, "2x2 / 3x3 / 5x5 / 7x7")

label_num = contar_objetos(otsu_eroded_image)
print(label_num)