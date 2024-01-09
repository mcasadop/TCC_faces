import time
import psutil
import matplotlib.pyplot as plt
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import cProfile
import pstats


def get_cpu_usage():
    return psutil.cpu_percent(interval=0.1)  # Intervalo de muestreo de 0.1 segundos

def main_process(tiempo_comienzo):
    archivos = [archivo for archivo in os.listdir(directorio_raiz)]
    contador = 1
    vectors = []
    uso_cpu = []
    time_data = []
    # DETECTAR CARA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device(process.py): {}'.format(device))
    mtcnn = MTCNN(
                select_largest=True,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
                image_size=160,
                device=device
            )
    encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()

    for imagen_nombre in archivos:
        try:
            elapsed_time = time.time() - tiempo_comienzo
            cpu_usage = get_cpu_usage()
            uso_cpu.append(cpu_usage)
            time_data.append(elapsed_time)  # Añadir el número de iteración a la lista de tiempo

            # Construir la ruta completa de la imagen
            ruta_imagen = os.path.join(directorio_raiz, imagen_nombre)

            # Abrir la imagen
            img = Image.open(ruta_imagen)

            cara = mtcnn(img)
            if cara is not None:  # Verificar si se detectó una cara
                embedding_cara = encoder.forward(cara.reshape((1, 3, 160, 160))).detach().cpu()
                vectors.append(embedding_cara)
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

    for i in range(len(vectors)):
        elapsed_time = time.time() - tiempo_comienzo
        cpu_usage = get_cpu_usage()
        uso_cpu.append(cpu_usage)
        # time_data.append(contador + i + 1)  # Incrementar el número de iteración
        time_data.append(elapsed_time)

        for j in range(i + 1, len(vectors)):
            distancia_euclidiana = np.linalg.norm(vectors[i] - vectors[j])
            print(f'Distancia entre vector {i+1} y vector {j+1}: {distancia_euclidiana}')
    print(len(vectors))
    return uso_cpu, time_data



if __name__ == "__main__":
    directorio_raiz = r'D:\Users\Miguel\Documents\TCC_faces\Primera\Input_Prueba'
    output_directory = r'D:\Users\Miguel\Documents\TCC_faces\Primera\output\Boxes'
    output_directory_vectors = r'D:\Users\Miguel\Documents\TCC_faces\Primera\output\Vectors'
    cpu_usage_data = []
    time_data = []
    start_time = time.time()
    print("Comienzo\n")
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # Llamar a la función principal
        cpu_usage_data, time_data = main_process(start_time)
        print(time.time() - start_time)
    except KeyboardInterrupt:
        pass

    finally:
        plt.plot(time_data, cpu_usage_data, label='Uso de CPU')  # Agregar 'marker' para marcar los puntos
        plt.xlabel('Tiempo de ejecución (s)')
        plt.ylabel('Uso de CPU (%)')
        plt.title('Rendimiento de la CPU durante la ejecución del proceso principal')
        plt.legend()
        plt.savefig(r'D:\Users\Miguel\Documents\TCC_faces\Primera\Output_Prueba\cpu_usage.png')
        # plt.show()
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(20)