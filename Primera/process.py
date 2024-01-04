from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import time

directorio_raiz = r'D:\Users\Miguel\Documents\TCC_faces\Primera\input\Humans'
output_directory = r'D:\Users\Miguel\Documents\TCC_faces\Primera\output\Boxes'
output_directory_vectors = r'D:\Users\Miguel\Documents\TCC_faces\Primera\output\Vectors'

archivos = [archivo for archivo in os.listdir(directorio_raiz) if archivo.endswith('.jpg')]

# Iniciar el cronómetro
tiempo_inicio = time.time()

# Inicializar el contador
contador = 1

# DETECTAR CARA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

for imagen_nombre in archivos:
    try:
        # Construir la ruta completa de la imagen
        ruta_imagen = os.path.join(directorio_raiz, imagen_nombre)

        # Abrir la imagen
        img = Image.open(ruta_imagen)

        mtcnn = MTCNN(
            select_largest=True,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            image_size=160,
            device=device
        )

        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        box = boxes[0] if boxes is not None else None
        landmark = landmarks[0] if landmarks is not None else None

        # Crear y guardar el gráfico si se detecta una cara
        if box is not None and landmark is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(img)
            ax.scatter(landmark[:, 0], landmark[:, 1], s=8, c='red')
            rect = plt.Rectangle(
                xy=(box[0], box[1]),
                width=box[2] - box[0],
                height=box[3] - box[1],
                fill=False,
                color='red'
            )
            ax.add_patch(rect)
            ax.axis('off')

            # Construir la ruta de guardado con el contador
            nombre_archivo_salida = f'output_{contador}.png'
            ruta_guardado = os.path.join(output_directory, nombre_archivo_salida)

            # Guardar el gráfico en la ubicación especificada
            plt.savefig(ruta_guardado)
            plt.close()  # Cerrar la figura para liberar memoria

            # VECTORIZAR
            encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()
            cara = mtcnn(img)
            if cara is not None:  # Verificar si se detectó una cara
                embedding_cara = encoder.forward(cara.reshape((1, 3, 160, 160))).detach().cpu()
                nombre_archivo_vector = f'vector_{contador}.txt'
                ruta_guardado_vector = os.path.join(output_directory_vectors, nombre_archivo_vector)

                with open(ruta_guardado_vector, 'w') as file:
                    for valor in embedding_cara.numpy().flatten():
                        file.write(f'{valor} ')
            
            # Incrementar el contador
            contador += 1

    except Exception as e:
        print(f'Error en la imagen {imagen_nombre}: {str(e)}')
        continue  # Continuar con la siguiente imagen en caso de error

# Detener el cronómetro
tiempo_fin = time.time()

# Calcular el tiempo total de ejecución
tiempo_total = tiempo_fin - tiempo_inicio
print(f'Tiempo total de ejecución: {tiempo_total:.2f} segundos')
