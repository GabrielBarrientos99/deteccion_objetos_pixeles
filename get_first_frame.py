import cv2

# Abrimos el video

video = cv2.VideoCapture('video_1_tarde.mp4')

# Leemos el primer frame
check, img = video.read()

if check:
    # Guardamos el primer frame
    cv2.imwrite('primer_frame.png', img)
    print('Primer frame guardado')
else:
    print('No se pudo leer el primer frame')

# liberamos el objeto de video
video.release()