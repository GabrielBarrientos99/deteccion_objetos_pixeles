import cv2
import pickle
import numpy as np
import os

# Cargar los datos de los estacionamientos desde el archivo 'espacios.pkl'
estacionamientos = []
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

# Leer el video
video = cv2.VideoCapture('video_1_tarde.mp4')

# Establecer el factor de salto de fotogramas para acelerar el video
skip_frames = 4  # Saltar 5 fotogramas para aumentar la velocidad

# Contador de objetos
objeto_contador = 0

# Bandera para detectar objetos
objeto_entrante = False
objeto_saliente = False

# Crear directorio para guardar imágenes si no existe
if not os.path.exists("detected_objects"):
    os.makedirs("detected_objects")

# Bucle para leer y procesar el video
while True:
    # Saltar fotogramas
    for _ in range(skip_frames):
        ret, img = video.read()

        # Si no hay más fotogramas, salir del bucle
        if not ret:
            break

    # Leer el siguiente fotograma después de saltar
    ret, img = video.read()

    if not ret:
        break

    # Escala de grises
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral adaptativo
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    # Aplicar filtro de mediana
    imgMedian = cv2.medianBlur(imgTH, 5)

    # Crear un kernel para la dilatación
    kernel = np.ones((5, 5), np.int8)

    # Dilatar las áreas o regiones de la imagen
    imgDil = cv2.dilate(imgMedian, kernel)

    # Dibujar los rectángulos en los estacionamientos
    for x, y, w, h in estacionamientos:
        espacio = imgDil[y:y+h, x:x+w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 10)

        # Mostrar el número de píxeles blancos dentro del cuadro
        cv2.putText(img, f"Blancos: {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)

        # Si el espacio tiene más de 900 píxeles no negros, marcarlo en verde
        if count > 900:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)
            objeto_entrante = True
            # Guardar la imagen con el objeto detectado (combinada con las sub-imágenes)
            img_filename = f"detected_objects/objeto_{objeto_contador}.png"
            cv2.imwrite(img_filename, combined_image)  # Guardar la imagen combinada
            print(f"Guardando imagen: {img_filename}")

        if objeto_entrante and count <= 0:
            objeto_saliente = True
            objeto_contador += 1
            objeto_entrante = False

    # Mostrar el contador en la imagen
    cv2.putText(img, f"Objetos detectados: {objeto_contador}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

    # Convertir las imágenes a tres canales (RGB/Color) para poder combinarlas
    imgBN_color = cv2.cvtColor(imgBN, cv2.COLOR_GRAY2BGR)
    imgTH_color = cv2.cvtColor(imgTH, cv2.COLOR_GRAY2BGR)
    imgMedian_color = cv2.cvtColor(imgMedian, cv2.COLOR_GRAY2BGR)
    imgDil_color = cv2.cvtColor(imgDil, cv2.COLOR_GRAY2BGR)

    # Redimensionar las imágenes para que tengan el mismo tamaño
    img_resized = cv2.resize(img, (300, 500))
    imgBN_resized = cv2.resize(imgBN_color, (300, 500))
    imgTH_resized = cv2.resize(imgTH_color, (300, 500))
    imgMedian_resized = cv2.resize(imgMedian_color, (300, 500))
    imgDil_resized = cv2.resize(imgDil_color, (300, 500))

    # Combinar las imágenes en una sola ventana (2x2)
    top_row = np.hstack((img_resized, imgBN_resized))  # Primera fila: Imagen original y escala de grises
    bottom_row = np.hstack((imgTH_resized, imgMedian_resized))  # Segunda fila: Umbral y mediana
    combined_image = np.vstack((top_row, bottom_row))  # Combinar ambas filas en una imagen de 4 subplots

    # Mostrar la imagen combinada
    cv2.imshow('Subplots', combined_image)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y cerrar ventanas
video.release()
cv2.destroyAllWindows()
