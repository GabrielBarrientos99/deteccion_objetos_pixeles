import cv2
import pickle
import numpy as np
import os

# Cargar los datos de los estacionamientos desde el archivo 'espacios.pkl'
detector_persona = []
with open('espacios.pkl', 'rb') as file:
    detector_persona = pickle.load(file)

# Leer los 3 videos (mañana,tarde y noche)
video1 = cv2.VideoCapture('video_1_tarde.mp4')
video2 = cv2.VideoCapture('video_2_tarde.mp4')
video3 = cv2.VideoCapture('video_3_tarde.mp4')

# Establecer el factor de salto de fotogramas para acelerar el video
skip_frames = 4  # Saltar 5 fotogramas para aumentar la velocidad

# Contador de objetos
objeto_contador1 = 0
objeto_contador2 = 0
objeto_contador3 = 0

# Bandera para detectar objetos
objeto_entrante1 = False
objeto_entrante2 = False
objeto_entrante3 = False

objeto_saliente1 = False
objeto_saliente2 = False
objeto_saliente3 = False

# Crear directorio para guardar imágenes para cada video si no existe
if not os.path.exists("detected_objects_video1"):
    os.makedirs("detected_objects_video1")
if not os.path.exists("detected_objects_video2"):
    os.makedirs("detected_objects_video2")
if not os.path.exists("detected_objects_video3"):
    os.makedirs("detected_objects_video3")


# Bucle para leer y procesar los 3 videos
while True:
    # Saltar fotogramas
    for _ in range(skip_frames):
        ret1, img1 = video1.read()
        ret2, img2 = video2.read()
        ret3, img3 = video3.read()


        # Si no hay más fotogramas, salir del bucle
        if not ret1 or not ret2 or not ret3:
            break

    # Leer el siguiente fotograma después de saltar para cada video
    ret1, img1 = video1.read()
    ret2, img2 = video2.read()
    ret3, img3 = video3.read()

    if not ret1 or not ret2 or not ret3:
        break

    # Escala de grises para el primer video
    imgBN1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Escala de grises para el segundo video
    imgBN2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Escala de grises para el tercer video
    imgBN3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)


    # Aplicar umbral adaptativo para el primer video
    imgTH1 = cv2.adaptiveThreshold(imgBN1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    # Aplicar umbral adaptativo para el segundo video
    imgTH2 = cv2.adaptiveThreshold(imgBN2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    # Aplicar umbral adaptativo para el tercer video
    imgTH3 = cv2.adaptiveThreshold(imgBN3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)



    # Aplicar filtro de mediana para el primer video
    imgMedian1 = cv2.medianBlur(imgTH1, 5)

    # Aplicar filtro de mediana para el segundo video
    imgMedian2 = cv2.medianBlur(imgTH2, 5)

    # Aplicar filtro de mediana para el tercer video
    imgMedian3 = cv2.medianBlur(imgTH3, 5)


    # Crear un kernel para la dilatación y dilatamos el video1
    kernel1 = np.ones((5, 5), np.int8)
    imgDil1 = cv2.dilate(imgMedian1, kernel1)
    
    # Crear un kernel para la dilatación y dilatamos el video2
    kernel2 = np.ones((5, 5), np.int8)
    imgDil2 = cv2.dilate(imgMedian2, kernel2)

    # Crear un kernel para la dilatación y dilatamos el video3
    kernel3 = np.ones((5, 5), np.int8)
    imgDil3 = cv2.dilate(imgMedian3, kernel3)


    # Dibujar los rectángulos en los estacionamientos
    for x, y, w, h in detector_persona:
        espacio1 = imgDil1[y:y+h, x:x+w]
        espacio2 = imgDil2[y:y+h, x:x+w]
        espacio3 = imgDil3[y:y+h, x:x+w]


        count1 = cv2.countNonZero(espacio1)
        count2 = cv2.countNonZero(espacio2)
        count3 = cv2.countNonZero(espacio3)


        cv2.putText(img1, str(count1), (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), 10)

        cv2.putText(img2, str(count2), (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 10)

        cv2.putText(img3, str(count3), (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)
        cv2.rectangle(img3, (x, y), (x+w, y+h), (255, 0, 0), 10)

        # Mostrar el número de píxeles blancos dentro del cuadro
        cv2.putText(img1, f"Blancos: {count1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)
        cv2.putText(img2, f"Blancos: {count2}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)
        cv2.putText(img3, f"Blancos: {count3}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 2)

        # Si el espacio tiene más de 900 píxeles no negros, marcarlo en verde
        if count1 > 900:
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 10)
            objeto_entrante1 = True
            # Guardar la imagen con el objeto detectado (combinada con las sub-imágenes)
            img_filename1 = f"detected_objects/objeto_{objeto_contador1}.png"
            cv2.imwrite(img_filename1, combined_image)  # Guardar la imagen combinada
            print(f"Guardando imagen: {img_filename1}")

        if objeto_entrante1 and count1 <= 0:
            objeto_saliente1 = True
            objeto_contador1 += 1
            objeto_entrante1 = False

    # Mostrar el contador en la imagen
    cv2.putText(img, f"Objetos detectados: {objeto_contador1}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

    # Convertir las imágenes a tres canales (RGB/Color) para poder combinarlas
    imgBN_color1 = cv2.cvtColor(imgBN1, cv2.COLOR_GRAY2BGR)
    imgTH_color1 = cv2.cvtColor(imgTH1, cv2.COLOR_GRAY2BGR)
    imgMedian_color1 = cv2.cvtColor(imgMedian1, cv2.COLOR_GRAY2BGR)
    imgDil_color1 = cv2.cvtColor(imgDil1, cv2.COLOR_GRAY2BGR)

    # Redimensionar las imágenes para que tengan el mismo tamaño
    img_resized1 = cv2.resize(img1, (300, 500))
    imgBN_resized1 = cv2.resize(imgBN_color1, (300, 500))
    imgTH_resized1 = cv2.resize(imgTH_color1, (300, 500))
    imgMedian_resized1 = cv2.resize(imgMedian_color1, (300, 500))
    imgDil_resized1 = cv2.resize(imgDil_color1, (300, 500))

    # Combinar las imágenes en una sola ventana (2x2)
    top_row1 = np.hstack((img_resized1, imgBN_resized1))  # Primera fila: Imagen original y escala de grises
    bottom_row1 = np.hstack((imgTH_resized1, imgMedian_resized1))  # Segunda fila: Umbral y mediana
    combined_image1 = np.vstack((top_row1, bottom_row1))  # Combinar ambas filas en una imagen de 4 subplots

    # Mostrar la imagen combinada
    cv2.imshow('Subplots', combined_image1)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y cerrar ventanas
video1.release()
video2.release()
video3.release()
cv2.destroyAllWindows()
