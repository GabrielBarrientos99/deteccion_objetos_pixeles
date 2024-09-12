import cv2
import pickle
import numpy as np
import os

# Cargar los datos de los estacionamientos desde el archivo 'espacios.pkl'
detector_persona = []
with open('espacios.pkl', 'rb') as file:
    detector_persona = pickle.load(file)

# Leer los 3 videos (mañana, tarde y noche)
video1 = cv2.VideoCapture('video_1_temprano.mp4')
video2 = cv2.VideoCapture('video_2_tarde.mp4')
video3 = cv2.VideoCapture('video_3_noche.mp4')

# Obtener la resolución del primer video
ret1, frame1 = video1.read()
height1, width1, _ = frame1.shape  # Obtener altura y anchura del primer video

# Establecer el factor de salto de fotogramas para acelerar el video
skip_frames = 4  # Saltar 4 fotogramas para aumentar la velocidad

# Contador de objetos detectados
objeto_contador1 = 0
objeto_contador2 = 0
objeto_contador3 = 0

# Referencia inicial de píxeles blancos (vacio)
ref_white_pixels1 = {}
ref_white_pixels2 = {}
ref_white_pixels3 = {}

# Bandera para detectar objetos
objeto_detectado1 = False
objeto_detectado2 = False
objeto_detectado3 = False

# Crear directorios para guardar imágenes de los tres videos
for i in range(1, 4):
    if not os.path.exists(f"detected_objects_video{i}"):
        os.makedirs(f"detected_objects_video{i}")

# Bucle para leer y procesar los 3 videos
while True:
    # Saltar fotogramas para cada video
    for _ in range(skip_frames):
        ret1, img1 = video1.read()
        ret2, img2 = video2.read()
        ret3, img3 = video3.read()

        if not ret1 or not ret2 or not ret3:
            break

    # Leer el siguiente fotograma después de saltar
    ret1, img1 = video1.read()
    ret2, img2 = video2.read()
    ret3, img3 = video3.read()

    if not ret1 or not ret2 or not ret3:
        break

    # Redimensionar los frames del segundo y tercer video para que coincidan con el tamaño del primero
    img2 = cv2.resize(img2, (width1, height1))
    img3 = cv2.resize(img3, (width1, height1))

    # Procesamiento para cada video
    def process_video(img, detector_persona, objeto_contador, ref_white_pixels, objeto_detectado, video_num):
        # Convertir a escala de grises
        imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Aplicar umbral adaptativo
        imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        # Aplicar filtro de mediana
        imgMedian = cv2.medianBlur(imgTH, 5)
        # Crear un kernel y aplicar dilatación
        kernel = np.ones((5, 5), np.int8)
        imgDil = cv2.dilate(imgMedian, kernel)

        # Detección de objetos
        for i, (x, y, w, h) in enumerate(detector_persona):
            # Aumentar el tamaño del recuadro para que sea más visible
            x = x - 10
            y = y - 10
            w = w + 20
            h = h + 20

            espacio = imgDil[y:y+h, x:x+w]
            count = cv2.countNonZero(espacio)

            # Inicializar la referencia de píxeles blancos si no está establecida
            if i not in ref_white_pixels:
                ref_white_pixels[i] = count

            # Mostrar número de píxeles blancos y el cuadro
            cv2.putText(img, f"Pixeles: {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)  # Recuadro más grueso

            # Detectar si el objeto ha entrado
            if count > ref_white_pixels[i] + 500:  # Ajuste de sensibilidad
                objeto_detectado = True
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Cuadro en verde más grueso
                
                # Guardar imagen del objeto detectado (cuando está dentro del recuadro)
                img_filename = f"detected_objects_video{video_num}/objeto_{objeto_contador + 1}.png"
                cv2.imwrite(img_filename, img)
                print(f"Guardando imagen: {img_filename}")

            # Si el objeto ha pasado (se reduce el número de píxeles a la referencia inicial)
            if objeto_detectado and count <= ref_white_pixels[i]:
                objeto_detectado = False
                objeto_contador += 1


            # Mostrar el contador de objetos detectados
            cv2.putText(img, f"Objetos: {objeto_contador}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return img, objeto_contador, objeto_detectado

    # Procesar cada video y contar objetos detectados
    img1, objeto_contador1, objeto_detectado1 = process_video(img1, detector_persona, objeto_contador1, ref_white_pixels1, objeto_detectado1, 1)
    img2, objeto_contador2, objeto_detectado2 = process_video(img2, detector_persona, objeto_contador2, ref_white_pixels2, objeto_detectado2, 2)
    img3, objeto_contador3, objeto_detectado3 = process_video(img3, detector_persona, objeto_contador3, ref_white_pixels3, objeto_detectado3, 3)

    # Redimensionar imágenes para mostrar en una sola ventana
    def resize_and_combine(img):
        img_resized = cv2.resize(img, (400, 600))  # Aumentar tamaño de las imágenes
        imgBN_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (400, 600))
        imgTH_resized = cv2.resize(cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16), (400, 600))
        imgMedian_resized = cv2.resize(cv2.medianBlur(imgTH_resized, 5), (400, 600))
        top_row = np.hstack((img_resized, cv2.cvtColor(imgBN_resized, cv2.COLOR_GRAY2BGR)))
        bottom_row = np.hstack((cv2.cvtColor(imgTH_resized, cv2.COLOR_GRAY2BGR), cv2.cvtColor(imgMedian_resized, cv2.COLOR_GRAY2BGR)))
        return np.vstack((top_row, bottom_row))

    # Combinar las imágenes de los tres videos en una ventana
    combined_image1 = resize_and_combine(img1)
    combined_image2 = resize_and_combine(img2)
    combined_image3 = resize_and_combine(img3)

    # Mostrar las imágenes combinadas para cada video
    cv2.imshow('Video 1', combined_image1)
    cv2.imshow('Video 2', combined_image2)
    cv2.imshow('Video 3', combined_image3)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los videos y cerrar ventanas
video1.release()
video2.release()
video3.release()
cv2.destroyAllWindows()
