import cv2
import pickle

# Leer una imagen
img = cv2.imread('fondo.png')

# Redimensionar la imagen para que la ventana de selección sea más pequeña
img_resized = cv2.resize(img, (300, 500))  # Cambia las dimensiones según tus necesidades

# Creando una lista para almacenar las áreas seleccionadas
espacios = []

for x in range(1):
    # Marcar un rectángulo en la imagen redimensionada
    espacio = cv2.selectROI('espacio', img_resized, False)
    cv2.destroyWindow('espacio')

    # Ajustar las coordenadas de la selección de ROI de vuelta al tamaño original (opcional)
    x, y, w, h = espacio
    x = int(x * (img.shape[1] / img_resized.shape[1]))
    y = int(y * (img.shape[0] / img_resized.shape[0]))
    w = int(w * (img.shape[1] / img_resized.shape[1]))
    h = int(h * (img.shape[0] / img_resized.shape[0]))

    # Añadir a la lista de espacios
    espacios.append((x, y, w, h))

    # Dibujar el rectángulo en la imagen original (no redimensionada)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Mostrar la imagen original con el rectángulo dibujado
cv2.imshow('Resultado', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar las coordenadas del rectángulo en un archivo pkl
with open('espacios.pkl', 'wb') as file:
    pickle.dump(espacios, file)
