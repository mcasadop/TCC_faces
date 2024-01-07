from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

from facenet_pytorch import MTCNN
import torch
import os
import time
import psutil

# DIRECTORIOS EDU
directorio_raiz = r'C:\Users\Eduardo\Downloads\PortableGit\TCC_faces\Primera\Input_Prueba'
output_directory = r'C:\Users\Eduardo\Downloads\PortableGit\TCC_faces\Primera\Output_Prueba\Boxes'
output_directory_vectors = r'C:\Users\Eduardo\Downloads\PortableGit\TCC_faces\Primera\Output_Prueba\Vectors'

archivos = [archivo for archivo in os.listdir(directorio_raiz) if archivo.endswith('.jpg')]


# Iniciar el cronómetro
tiempo_inicio = time.time()

# Inicializar el contador
contador = 1

# DETECTAR CARA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: (process_M1) {}'.format(device))

# Lista para almacenar el uso de la CPU a lo largo del tiempo
cpu_percent_history = []

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

            # VECTORIZAR con NumPy (reemplazando InceptionResnetV1)
            cara_np = np.array(img)  # Convertir imagen a arreglo NumPy
            cara_flat = cara_np.flatten()  # Aplanar el arreglo

            # Guardar el vector en un archivo de texto
            nombre_archivo_vector = f'vector_{contador}.txt'
            ruta_guardado_vector = os.path.join(output_directory_vectors, nombre_archivo_vector)

            with open(ruta_guardado_vector, 'w') as file:
                for valor in cara_flat:
                    file.write(f'{valor} ')

            # Incrementar el contador
            contador += 1

    except Exception as e:
        print(f'Error en la imagen {imagen_nombre}: {str(e)}')
        continue  # Continuar con la siguiente imagen en caso de error

    # Registrar el uso de la CPU cada segundo
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_percent_history.append(cpu_percent)

# Detener el cronómetro
tiempo_fin = time.time()

# Calcular el tiempo total de ejecución
tiempo_total = tiempo_fin - tiempo_inicio
print(f'Tiempo total de ejecución: {tiempo_total:.2f} segundos')

# Crear un gráfico de líneas para el uso de CPU a lo largo del tiempo
tiempo_history = list(range(len(cpu_percent_history)))
plt.plot(tiempo_history, cpu_percent_history, label='Uso de CPU', color='tab:blue', marker='o')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Uso de CPU (%)')
plt.title('Tendencia del Uso de CPU a lo largo del Tiempo')
plt.legend()
plt.show()