import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from PIL import Image

# Chargement de l'image
image = cv2.imread('hips.jpg')

#----------------------------------------------------------------------------------------conversion en niveaux de gris et normalisation de l'image
# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normaliser l'image en niveaux de gris
normalized_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

#------------------------------------------------------------------------------------------ Contraste, luminosité et égalisation
# Ajuster automatiquement la luminosité et le contraste
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
adjusted_auto_gray = clahe.apply(gray)

# Appliquer l'égalisation d'histogramme
equalized_image = cv2.equalizeHist(gray)

# Afficher les images ajustées automatiquement
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image originale')

plt.subplot(1, 3, 2)
plt.imshow(adjusted_auto_gray, cmap='gray')
plt.title('Image ajustée (contraste et luminosité)')

plt.subplot(1, 3, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Image égalisée')
#------------------------------------------------------------------------------------------------ Traitements de l'image 

# Appliquer un bruit gaussien à l'image en niveaux de gris
noise_gray = np.random.normal(0, 25, gray.shape)
noisy_image_gray = cv2.add(gray, noise_gray.astype(np.uint8))

# Appliquer un bruit gaussien à l'image normalisée
noise_normalized = np.random.normal(0, 25, normalized_gray.shape)
noisy_image_normalized = cv2.add(normalized_gray, noise_normalized.astype(np.uint8))

# Appliquer un flou pour réduire le bruit
blurred_gray = cv2.GaussianBlur(noisy_image_gray, (5, 5), 0)
blurred_normalized = cv2.GaussianBlur(noisy_image_normalized, (5, 5), 0)

# Appliquer le débruitage avec le filtre NLMeans
denoised_gray = cv2.fastNlMeansDenoising(blurred_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
denoised_normalized = cv2.fastNlMeansDenoising(blurred_normalized, None, h=10, templateWindowSize=7, searchWindowSize=21)


#--------------------------------------------------- contours 

# Appliquer la détection des contours avec Canny
edges_canny = cv2.Canny(blurred_gray, 50, 200)

# Afficher les contours
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image originale')

plt.subplot(1, 2, 2)
plt.imshow(edges_canny, cmap='gray')
plt.title('Contours de l''image')

#------------------------------------- sobel
# Appliquer le filtre de Sobel sur l'image en niveaux de gris
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convertir les résultats de Sobel en valeurs absolues
sobelx = np.absolute(sobelx)
sobely = np.absolute(sobely)

# Convertir les résultats en entiers 8 bits
sobelx = np.uint8(sobelx)
sobely = np.uint8(sobely)

# Combinez les résultats de Sobel
sobel_combined = cv2.bitwise_or(sobelx, sobely)

# Afficher les contours
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image originale')

plt.subplot(1, 2, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Image avec filtres de Sobel')

#---------Fin Sobbel

# Appliquer l'érosion
kernel_erosion = np.ones((5, 5), np.uint8)
eroded = cv2.erode(gray, kernel_erosion, iterations=1)

# Appliquer la dilatation
kernel_dilation = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(gray, kernel_dilation, iterations=1)

# Affichage des images
plt.figure(figsize=(10, 10))

# Image originale
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image originale')

# Image en niveaux de gris
plt.subplot(2, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Image en niveaux de gris')

# Image normzlisée en niveaux de gris
plt.subplot(2, 4, 3)
plt.imshow(normalized_gray)
plt.title('Image normalisée en niveaux de gris')

# Image bruitée
plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(noisy_image_gray, cv2.COLOR_BGR2RGB))
plt.title('Image bruitée')

# Image débruitée
plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(denoised_gray, cv2.COLOR_BGR2RGB))
plt.title('Image débruitée')

# Imahe floutée
plt.subplot(2, 4, 6)
plt.imshow(blurred_gray, cmap='gray')
plt.title('Image floutée')

# Image érodée
plt.subplot(2, 4, 7)
plt.imshow(eroded, cmap='gray')
plt.title('Image érodée')

# Image dilatée
plt.subplot(2, 4, 8)
plt.imshow(dilated, cmap='gray')
plt.title('Image dilatée')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
#------------------------------------------------------------------------------------------------------- Histogramme
# Calculer l'histogramme de l'image en niveaux de gris
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Trouver les valeurs minimales et maximales de l'histogramme
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)

# Afficher les histogrammes dans une autre figure
plt.figure(figsize=(10, 7))

# Histogramme de l'image originale
plt.subplot(2, 3, 1)
plt.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]))
plt.title('Histogramme original')

# Histogramme de l'image originale
plt.subplot(2, 3, 2)
plt.plot(cv2.calcHist([equalized_image], [0], None, [256], [0, 256]))
plt.title('Histogramme égalisée')

# Histogramme de l'image bruitée
plt.subplot(2, 3, 3)
plt.plot(cv2.calcHist([blurred_gray], [0], None, [256], [0, 256]))
plt.title('Histogramme bruité')

# Histogramme de l'image débruitée
plt.subplot(2, 3, 4)
plt.plot(cv2.calcHist([denoised_gray], [0], None, [256], [0, 256]))
plt.title('Histogramme débruité')

# Histogramme de l'image érodée
plt.subplot(2, 3, 5)
plt.plot(cv2.calcHist([eroded], [0], None, [256], [0, 256]))
plt.title('Histogramme érodé')


# Histogramme de l'image érodée
plt.subplot(2, 3, 6)
plt.plot(cv2.calcHist([dilated], [0], None, [256], [0, 256]))
plt.title('Histogramme dilaté')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
#-------------------------------------------------------------------------------------------------------TTF et TTF-1
# Calculer la transformée de Fourier
fft_image = np.fft.fft2(blurred_gray)
fft_shifted = np.fft.fftshift(fft_image)
magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)  # Log pour mieux visualiser

# Calculer la transformée inverse de Fourier
ifft_shifted = np.fft.ifftshift(fft_shifted)
ifft_image = np.fft.ifft2(ifft_shifted)
ifft_image = np.abs(ifft_image)

# Afficher toutes les images dans une figure : TTF
plt.figure(figsize=(10, 5))

# Transformée de Fourier
plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Transformée de Fourier')

# Transformée inverse de Fourier
plt.subplot(1, 2, 2)
plt.imshow(ifft_image, cmap='gray')
plt.title('Transformée Inverse de Fourier')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
#------------------------------------------------------------------------------------------------ Entropie
# Calculer l'entropie des images
entropy_original = entropy(image.ravel())
entropy_gray = entropy(gray.ravel())
entropy_noisy = entropy(noisy_image_gray.ravel())
entropy_denoised = entropy(denoised_gray.ravel())
entropy_blurred = entropy(blurred_gray.ravel())

# Imprimer les entropies calculées
print(f"Entropie de l'image originale : {entropy_original}")
print(f"Entropie de l'image en niveaux de gris : {entropy_gray}")
print(f"Entropie de l'image bruitee : {entropy_noisy}")
print(f"Entropie de l'image debruitee : {entropy_denoised}")
print(f"Entropie de l'image floue : {entropy_blurred}")

plt.figure(figsize=(10, 10))

# Entropie Image originale
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Image originale - Entropie: {entropy_original:.2f}')

# Entropie Image en niveaux de gris
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')  # Affiche l'image en niveaux de gris
plt.title(f'Image en niveaux de gris - Entropie: {entropy_gray:.2f}')

# Entropie Image bruitée
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(noisy_image_gray, cv2.COLOR_BGR2RGB))
plt.title(f'Image bruitée - Entropie: {entropy_noisy:.2f}')

# Entropie Image débruitée
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(denoised_gray, cv2.COLOR_BGR2RGB))
plt.title(f'Image débruitée - Entropie: {entropy_denoised:.2f}')

# Entropie Image floue
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(denoised_gray, cv2.COLOR_BGR2RGB))
plt.title(f'Image floue - Entropie : {entropy_blurred:.2f}')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
#--------------------------------------------------------------------------------------------------- KMeans
# Appliquer K-Means
k = 5  # Nombre de clusters
data = gray.reshape((-1, 1))  # Utiliser l'image en niveaux de gris
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir les centres en entiers
centers = np.uint8(centers)

# Recréer l'image segmentée
segmented = centers[labels.flatten()]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image originale')

plt.subplot(1, 2, 2)
plt.imshow(segmented.reshape(gray.shape), cmap='gray')
plt.title('Image segmentée (K-Means)')

plt.show()