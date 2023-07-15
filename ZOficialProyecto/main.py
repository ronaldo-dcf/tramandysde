import cv2
import numpy as np

imageprueba1 = cv2.imread("puente-americas-1.jpg")
imageprueba2 = cv2.imread("puente-americas-2.jpg")

cv2.imshow("Primera Imagen", imageprueba1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Segunda Imagen", imageprueba2)
cv2.waitKey(0)
cv2.destroyAllWindows()

graycolores1 = cv2.cvtColor(imageprueba1, cv2.COLOR_BGR2GRAY)
graycolores2 = cv2.cvtColor(imageprueba2, cv2.COLOR_BGR2GRAY)

tamorb = cv2.ORB_create()

keypoints1, descriptors1 = tamorb.detectAndCompute(graycolores1, None)
keypoints2, descriptors2 = tamorb.detectAndCompute(graycolores2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

distanciamatches = bf.match(descriptors1, descriptors2)

distanciamatches = sorted(distanciamatches, key=lambda x: x.distance)

pts1 = np.float32([keypoints1[m.queryIdx].pt for m in distanciamatches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in distanciamatches]).reshape(-1, 1, 2)

H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

h, w = graycolores1.shape

transformed_image = cv2.warpPerspective(imageprueba2, H, (w * 2, h))

panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
panorama[:, :w] = imageprueba1
panorama[:, w:] = transformed_image[:, w:]

overlap_width = 100
panorama = panorama[:, :-overlap_width]

result = cv2.drawMatches(imageprueba1, keypoints1, imageprueba2, keypoints2, distanciamatches[:10], None, flags=2)
cv2.imshow("Coincidencias", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Imagen Unida Panoramica", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

x1, y1 = pts1[0][0]
x2, y2 = pts2[0][0]

print("Coordenadas de los puntos de coincidencia:")
print("Punto 1:", x1, y1)
print("Punto 2:", x2, y2)

inicio_recorte_y = int(max(y1, y2))
#inicio_recorte_y = 180
fin_recorte_y = panorama.shape[0]
anchura_recortada = int(max(x1, x2))
altura_recortada = panorama.shape[0] - inicio_recorte_y
#altura_recortada = panorama.shape[0] - 280

panorama_recortada = panorama[inicio_recorte_y:fin_recorte_y, :anchura_recortada]

cv2.imshow("Panorama Recortada", panorama_recortada)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
import cv2
import numpy as np


imageprueba1 = cv2.imread("puente-americas-1.jpg")
imageprueba2 = cv2.imread("puente-americas-2.jpg")

# Se muestra que imagen es en cada uno son imagenes de una fotografia de una paisaje
cv2.imshow("Primera Imagen", imageprueba1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Segunda Imagen", imageprueba2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#se convierte a grises
graycolores1 = cv2.cvtColor(imageprueba1, cv2.COLOR_BGR2GRAY)
graycolores2 = cv2.cvtColor(imageprueba2, cv2.COLOR_BGR2GRAY)


tamorb = cv2.ORB_create()


keypoints1, descriptors1 = tamorb.detectAndCompute(graycolores1, None)
keypoints2, descriptors2 = tamorb.detectAndCompute(graycolores2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


distanciamatches = bf.match(descriptors1, descriptors2)


distanciamatches = sorted(distanciamatches, key=lambda x: x.distance)

# Obtener los puntos de las imágenes
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in distanciamatches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in distanciamatches]).reshape(-1, 1, 2)

# Calculando la matriz
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# El resultado de las imagenes
h, w = graycolores1.shape

# Tranformación de perpectiva 
transformed_image = cv2.warpPerspective(imageprueba2, H, (w * 2, h))

# Combinar imágenes y recortar el área superpuesta
panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
panorama[:, :w] = imageprueba1
panorama[:, w:] = transformed_image[:, w:]

# Area donde se superponen las imágenes
overlap_width = 100
panorama = panorama[:, :-overlap_width]

# Coincidencias en una imagen
result = cv2.drawMatches(imageprueba1, keypoints1, imageprueba2, keypoints2, distanciamatches[:10], None, flags=2)
cv2.imshow("Coincidencias", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen panorámica
cv2.imshow("Imagen Unida Panoramica", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Coordenadas de la intersección de las imágenes
x1, y1 = pts1[0][0]
x2, y2 = pts2[0][0]

print(x1,y1)
print(x2,y2)

inicio_recorte_y = 180
fin_recorte_y = panorama.shape[1]  
print(panorama.shape[2] )
print(panorama.shape[1] )
anchura_recortada = panorama.shape[1] - 280

altura_recortada = fin_recorte_y - inicio_recorte_y


panorama_recortada = panorama[inicio_recorte_y:fin_recorte_y, :anchura_recortada]


# Imagen panorámica recortada
cv2.imshow("Panorama Recortada", panorama_recortada)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""







"""
import cv2
import numpy as np

# Cargar las imágenes
image1 = cv2.imread("puente-americas-1.jpg")
image2 = cv2.imread("puente-americas-2.jpg")

# Mostrar la imagen 1
cv2.imshow("Imagen 1", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen 2
cv2.imshow("Imagen 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Crear el detector ORB
orb = cv2.ORB_create()

# Encontrar los puntos clave y descriptores en las imágenes
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Crear el objeto BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Realizar el emparejamiento de características
matches = bf.match(descriptors1, descriptors2)

# Ordenar las coincidencias por distancia
matches = sorted(matches, key=lambda x: x.distance)

# Obtener los puntos correspondientes en ambas imágenes
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Calcular la matriz de transformación de perspectiva
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen resultante
h, w = gray1.shape

# Realizar la transformación de perspectiva en la imagen 2
transformed_image = cv2.warpPerspective(image2, H, (w * 2, h))

# Combinar las imágenes y recortar el área superpuesta
panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
panorama[:, :w] = image1
panorama[:, w:] = transformed_image[:, w:]

# Recortar el área donde se superponen las imágenes
overlap_width = 100
panorama = panorama[:, :-overlap_width]

# Mostrar las coincidencias en una imagen
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=2)
cv2.imshow("Coincidencias", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen panorámica
cv2.imshow("Imagen Panorámica", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Coordenadas del área de recorte en el eje x
inicio_recorte_x = int(w // w)  # Empieza en la mitad de la imagen
fin_recorte_x = panorama.shape[1] - 290  # Termina 30 píxeles antes del final de la imagen panorámica

# Coordenadas del área de recorte en el eje y
inicio_recorte_y = int(h * 0.4)  # Recorta el 20% desde la parte superior de la imagen
fin_recorte_y = panorama.shape[0]  # Termina al final de la imagen panorámica

# Recortar la imagen panorámica
panorama_recortada = panorama[inicio_recorte_y:fin_recorte_y, inicio_recorte_x:fin_recorte_x]

# Mostrar la imagen panorámica recortada
cv2.imshow("Panorama Recortada", panorama_recortada)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""




"""
import cv2
import numpy as np

# Cargar las imágenes
image1 = cv2.imread("puente-americas-1.jpg")
image2 = cv2.imread("puente-americas-2.jpg")

# Mostrar la imagen 1
cv2.imshow("Imagen 1", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen 2
cv2.imshow("Imagen 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Crear el detector ORB
orb = cv2.ORB_create()

# Encontrar los puntos clave y descriptores en las imágenes
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Crear el objeto BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Realizar el emparejamiento de características
matches = bf.match(descriptors1, descriptors2)

# Ordenar las coincidencias por distancia
matches = sorted(matches, key=lambda x: x.distance)

# Obtener los puntos correspondientes en ambas imágenes
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Calcular la matriz de transformación de perspectiva
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen resultante
h, w = gray1.shape

# Realizar la transformación de perspectiva en la imagen 2
transformed_image = cv2.warpPerspective(image2, H, (w * 2, h))

# Combinar las imágenes y recortar el área superpuesta
panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
panorama[:, :w] = image1
panorama[:, w:] = transformed_image[:, w:]

# Recortar el área donde se superponen las imágenes
overlap_width = 100
panorama = panorama[:, :-overlap_width]

# Mostrar las coincidencias en una imagen
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=2)
cv2.imshow("Coincidencias", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen panorámica
cv2.imshow("Imagen Panorámica", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""




"""
import cv2
import os

mainFolder = 'Images'
myFolders = os.listdir(mainFolder)
print(myFolders)

for folder in myFolders:
    path = mainFolder +'/'+folder
    images =[]
    myList = os.listdir(path)
    print(f'Total no of images detected {len(myList)}')
    for imgN in myList:
        curImg = cv2.imread(f'{path}/{imgN}')
        curImg = cv2.resize(curImg,(0,0),None,0.2,0.2)
        images.append(curImg)

    stitcher = cv2.Stitcher.create()
    (status,result) = stitcher.stitch(images)
    if(status == cv2.STITCHER_OK):
        print('Panorama Generated')
        cv2.imshow(folder,result)
        cv2.waitKey(1)
    else:
        print('panorama generation Unsuccessful')

cv2.waitKey(0)
"""  


"""
import cv2
import numpy as np

imageA = cv2.imread("puente-americas-1.jpg")
imageB = cv2.imread("puente-americas-2.jpg")

if imageA is None or imageB is None:
    print("No image data.")
    cv2.waitKey(0)
    exit(-1)

detect = cv2.BRISK_create()

kpA, descA = detect.detectAndCompute(imageA, None)
kpB, descB = detect.detectAndCompute(imageB, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descA, descB)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:30]

result = cv2.drawMatches(imageA, kpA, imageB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

pts1 = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

h, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

print("Homography Mat:")
print(h)

box = np.array([
    [0, 0],
    [imageB.shape[1], 0],
    [imageB.shape[1], imageB.shape[0]],
    [0, imageB.shape[0]]
], dtype=np.float32)

box_dst = cv2.perspectiveTransform(box.reshape(-1, 1, 2), h)

rc = cv2.boundingRect(box_dst)

dst = cv2.warpPerspective(imageB, h, (rc[2] + rc[0], rc[3] + rc[1]))
dst[rc[1]:rc[1] + imageA.shape[0], rc[0]:rc[0] + imageA.shape[1]] = imageA.copy()

cv2.imshow("Puente de las Americas", dst)
cv2.imshow("OpenCV :: Match Keypoints", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




"""
import cv2
import numpy as np

imageA = cv2.imread("puente-americas-1.jpg")
imageB = cv2.imread("puente-americas-2.jpg")

if imageA is None or imageB is None:
    print("No image data.")
    cv2.waitKey(0)
    exit(-1)

detect = cv2.BRISK_create()

kpA, descA = detect.detectAndCompute(imageA, None)
kpB, descB = detect.detectAndCompute(imageB, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descA, descB)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:30]

result = cv2.drawMatches(imageA, kpA, imageB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

pts1 = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

h, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

print("Homography Mat:")
print(h)

box = np.float32([[0, 0], [imageB.shape[1], 0], [imageB.shape[1], imageB.shape[0]], [0, imageB.shape[0]]])
box_dst = cv2.perspectiveTransform(box.reshape(-1, 1, 2), h)

rc = cv2.boundingRect(box_dst)

dst = cv2.warpPerspective(imageB, h, (rc[2] + rc[0], rc[3] + rc[1]))
dst[rc[1]:rc[1] + imageA.shape[0], rc[0]:rc[0] + imageA.shape[1]] = imageA.copy()

cv2.imshow("Puenta de las Americas", dst)
cv2.imshow("OpenCV :: Match Keypoints", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




"""
import cv2
import numpy as np

imageA = cv2.imread("puente-americas-1.jpg")
imageB = cv2.imread("puente-americas-2.jpg")

if imageA is None or imageB is None:
    print("No image data.")
    cv2.waitKey(0)
    exit(-1)

detect = cv2.BRISK_create()

kpA, descA = detect.detectAndCompute(imageA, None)
kpB, descB = detect.detectAndCompute(imageB, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descA, descB)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:30]

result = cv2.drawMatches(imageA, kpA, imageB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

pts1 = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

h, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

# Ajustar la posición de la imagen B hacia abajo
h_shift = np.array([[1, 0, 0], [0, 1, 100], [0, 0, 1]])
h = np.dot(h_shift, h)

box = np.float32([[0, 0], [imageB.shape[1], 0], [imageB.shape[1], imageB.shape[0]], [0, imageB.shape[0]]])
box_dst = cv2.perspectiveTransform(box.reshape(-1, 1, 2), h)

rc = cv2.boundingRect(box_dst)
dst = cv2.warpPerspective(imageB, h, (rc[2] + rc[0], rc[3] + rc[1]))
dst[rc[1]:rc[1] + imageA.shape[0], rc[0]:rc[0] + imageA.shape[1]] = imageA

# Dibujar las líneas que conectan los puntos correspondientes
for match in matches:
    pt1 = tuple(map(int, kpA[match.queryIdx].pt))
    pt2 = tuple(map(int, kpB[match.trainIdx].pt))
    cv2.line(result, pt1, pt2, (0, 255, 0), 1)

cv2.imshow("Puente de las Americas", dst)
cv2.imshow("OpenCV :: Match Keypoints", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""






"""
import cv2
import numpy as np

imageA = cv2.imread("puente-americas-1.jpg")
imageB = cv2.imread("puente-americas-2.jpg")

if imageA is None or imageB is None:
    print("No image data.")
    cv2.waitKey(0)
    exit(-1)

detect = cv2.BRISK_create()

kpA, descA = detect.detectAndCompute(imageA, None)
kpB, descB = detect.detectAndCompute(imageB, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descA, descB)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:30]

result = cv2.drawMatches(imageA, kpA, imageB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

pts1 = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

h, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

box = np.float32([[0, 0], [imageB.shape[1], 0], [imageB.shape[1], imageB.shape[0]], [0, imageB.shape[0]]])
box_dst = cv2.perspectiveTransform(box.reshape(-1, 1, 2), h)

rc = cv2.boundingRect(box_dst)
dst = cv2.warpPerspective(imageB, h, (rc[2] + rc[0], rc[3] + rc[1]))
dst[rc[1]:rc[1] + imageA.shape[0], rc[0]:rc[0] + imageA.shape[1]] = imageA

cv2.imshow("Puente de las Americas", dst)
cv2.imshow("OpenCV :: Match Keypoints", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



"""
import cv2
import numpy as np

# Cargar las dos partes de la imagen
image1 = cv2.imread("puente-americas-1.jpg")
image2 = cv2.imread("puente-americas-2.jpg")

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Inicializar el detector y descriptor de características
orb = cv2.ORB_create()

# Encontrar los puntos clave y descriptores en las imágenes
kp1, desc1 = orb.detectAndCompute(gray1, None)
kp2, desc2 = orb.detectAndCompute(gray2, None)

# Encontrar las coincidencias entre los descriptores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

# Seleccionar las mejores coincidencias
num_matches = 10  # Selecciona un número adecuado de coincidencias
best_matches = matches[:num_matches]

# Obtener los puntos correspondientes en ambas imágenes
pts1 = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Calcular la matriz de transformación de perspectiva que alinea la imagen 2 con la imagen 1
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen 1
h, w = image1.shape[:2]

# Aplicar la matriz de transformación a la imagen 2 para alinearla con la imagen 1
image2_aligned = cv2.warpPerspective(image2, M, (w, h))

# Combinar las imágenes para formar una imagen panorámica
panorama = np.maximum(image1, image2_aligned)

# Mostrar las coincidencias entre las imágenes
image_matches = cv2.drawMatches(image1, kp1, image2_aligned, kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Coincidencias", image_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen panorámica
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




"""
import cv2
import numpy as np

# Cargar las dos partes de la imagen
image1 = cv2.imread("puente-americas-1.jpg")
image2 = cv2.imread("puente-americas-2.jpg")

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Inicializar el detector y descriptor de características
orb = cv2.ORB_create()

# Encontrar los puntos clave y descriptores en las imágenes
kp1, desc1 = orb.detectAndCompute(gray1, None)
kp2, desc2 = orb.detectAndCompute(gray2, None)

# Encontrar las coincidencias entre los descriptores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

# Seleccionar las mejores coincidencias
num_matches = 10  # Selecciona un número adecuado de coincidencias
best_matches = matches[:num_matches]

# Obtener los puntos correspondientes en ambas imágenes
pts1 = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Estimar la matriz de transformación de perspectiva
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen resultante
h, w = image1.shape[:2]

# Superponer la imagen 2 en la imagen 1 utilizando la matriz de transformación
result = cv2.warpPerspective(image2, M, (w, h))

# Obtener una máscara de los píxeles no superpuestos
mask = np.zeros_like(image1, dtype=np.uint8)
cv2.fillConvexPoly(mask, pts1.astype(int), (255, 255, 255))
mask_inv = cv2.bitwise_not(mask)

# Combinar las imágenes utilizando la máscara
masked_image1 = cv2.bitwise_and(image1, mask_inv)
final_image = cv2.add(masked_image1, result)

# Mostrar la imagen 1
cv2.imshow("Imagen 1", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen resultante
cv2.imshow("Imagen 2 Complementada", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
import cv2
import numpy as np

# Cargar las dos partes de la imagen
image1 = cv2.imread("puente-americas-1.jpg")
image2 = cv2.imread("puente-americas-2.jpg")

# Mostrar la imagen 1
cv2.imshow("Imagen 1", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar la imagen 2
cv2.imshow("Imagen 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Inicializar el detector y descriptor de características
orb = cv2.ORB_create()

# Encontrar los puntos clave y descriptores en las imágenes
kp1, desc1 = orb.detectAndCompute(gray1, None)
kp2, desc2 = orb.detectAndCompute(gray2, None)

# Encontrar las coincidencias entre los descriptores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

# Seleccionar las mejores coincidencias
num_matches = 10  # Selecciona un número adecuado de coincidencias
best_matches = matches[:num_matches]

# Obtener los puntos correspondientes en ambas imágenes
pts1 = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Estimar la matriz de transformación de perspectiva
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen resultante
h, w = image1.shape[:2]

# Superponer la imagen 2 en la imagen 1 utilizando la matriz de transformación
result = cv2.warpPerspective(image2, M, (w, h))

# Obtener una máscara de los píxeles no superpuestos
mask = np.zeros_like(image1, dtype=np.uint8)
cv2.fillConvexPoly(mask, pts1.astype(int), (255, 255, 255))
mask_inv = cv2.bitwise_not(mask)

# Combinar las imágenes utilizando la máscara
masked_image1 = cv2.bitwise_and(image1, mask_inv)
final_image = cv2.add(masked_image1, result)

# Dibujar las líneas que conectan los puntos correspondientes
image_matches = cv2.drawMatches(image1, kp1, image2, kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
for match in best_matches:
    pt1 = tuple(map(int, kp1[match.queryIdx].pt))
    pt2 = tuple(map(int, kp2[match.trainIdx].pt))
    cv2.line(image_matches, pt1, pt2, (0, 255, 0), 1)

# Mostrar la imagen resultante con las líneas
cv2.imshow("Coincidencias", image_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



"""

import cv2
import numpy as np

# Cargar las dos partes de la imagen
image1 = cv2.imread("puente-americas-1.jpg")
image2 = cv2.imread("puente-americas-2.jpg")

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Inicializar el detector y descriptor de características
orb = cv2.ORB_create()

# Encontrar los puntos clave y descriptores en las imágenes
kp1, desc1 = orb.detectAndCompute(gray1, None)
kp2, desc2 = orb.detectAndCompute(gray2, None)

# Encontrar las coincidencias entre los descriptores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

# Seleccionar las mejores coincidencias
num_matches = 10  # Selecciona un número adecuado de coincidencias
best_matches = matches[:num_matches]

# Obtener los puntos correspondientes en ambas imágenes
pts1 = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Calcular la transformación de perspectiva que alinea la imagen 2 con la imagen 1
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen 1
h, w = image1.shape[:2]

# Aplicar la transformación de perspectiva a la imagen 2
image2_aligned = cv2.warpPerspective(image2, M, (w, h))

# Superponer las imágenes en las áreas de coincidencia
image_blend = cv2.addWeighted(image1, 0.5, image2_aligned, 0.5, 0.0)

# Mostrar la imagen resultante
cv2.imshow("Panorama", image_blend)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
