# CODIGO PARA PROYECTO DE PROCESAMIENTO DIGITAL DE SEÑALES E IMAGENES
# MAIN
import numpy as np
import cv2 as cv
from ultralytics import YOLO

# Ruta del video y modelo
video_path = r'D:\VSCode\PROCESAMIENTO DIGITAL DE SEÑALES E IMÁGENES\VIDEO_8.mp4'
model_path = r'best.pt'

# Cargamos el modelo YOLO
model = YOLO(model_path)
#print(model.names)  -> Clases del modelo

# Clases a etiquetar
# Etiquetas para renombrar
class_mapping = {0: 'vehiculo', 2: 'peaton'}

# Puntos donde se ubicará el polígono (ROI)
pts = np.array([[430, 1079], [1740, 1079], [1155, 680], [835, 680]], np.int32)
pts = pts.reshape((-1, 1, 2))
# Colores
verde = (0, 255, 102)  # Verde neón
rojo = (0, 102, 255)  # Rojo neón

# Cargamos el video
cap = cv.VideoCapture(video_path)
font = cv.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    # Aseguramos que se lean los frames
    if not ret:
        break

    # bandera para saber si hay algún objeto dentro
    dentro = False

    # Pasamos los frames para la predicción
    results = model(frame, stream=True)

    for result in results:
        # result es el objeto con la información de los frames detectados
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # recorremos las detecciones en el frame
            x1, y1, x2, y2 = map(int, box)
            label = class_mapping.get(int(class_id), f'Clase {int(class_id)}')  
            conf = f'{confidence:.2f}' # mostramos solo con 2 decimales

            # Coordenadas 
            # Para saber si la detección se ha dado, tomaremos 5 puntos. Las 4 esquinas
            # y el centro del borde inferior de
            ix = (x1 + x2) // 2 
            iy = y2
            points = [ (x1, y2), (x2, y2), (ix , iy)]
            
            for point in points:
                if cv.pointPolygonTest(pts, point, False) >= 0:  # Verificamos si la esquina está dentro del polígono
                    dentro = True
                    break

            # Si está dentro del polígono, cambiar la caja a rojo, si no, verde
            color = rojo if dentro else verde

            # Dibujar la caja y la etiqueta
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f'{label} {conf}', (x1, y1 - 10), font, 1, (0, 0, 255), 1, cv.LINE_AA)

    # Cambiar el color del borde y la máscara del polígono (ROI)
    mask = np.zeros_like(frame)
    if dentro:
        # Máscara roja
        cv.fillPoly(mask, [pts], rojo)  
        cv.polylines(frame, [pts], True, rojo, 2, cv.LINE_AA) 
        cv.putText(frame,'FRENAR!',(910,750), font, 1.5,(255,255,255),2,cv.LINE_AA)
    else:
        # Máscara verde
        cv.fillPoly(mask, [pts], verde)  
        cv.polylines(frame, [pts], True, verde, 2, cv.LINE_AA) 

    # Combinamos la imagen con la máscara (para mostrar la región de interés con color actualizado)
    alpha = 0.25  # Grado de transparencia
    frame = cv.addWeighted(mask, alpha, frame, 1, 0)

    # Mostramos los frames
    cv.imshow('PDSI-DETECTION', frame)

    # Esperamos la tecla 'c' para salir
    if cv.waitKey(1) & 0xFF == 99:
        break

cap.release()
cv.destroyAllWindows()
