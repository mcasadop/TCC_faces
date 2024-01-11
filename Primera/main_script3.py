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
import yaml

def get_cpu_usage():
    return psutil.cpu_percent(interval=0.1)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def main_process(tiempo_comienzo):
    archivos = [archivo for archivo in os.listdir(directorio_raiz)]
    contador = 1
    vectors = []
    uso_cpu = []
    uso_memoria = []
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
        print(f'Procesando: {imagen_nombre}')
        try:
            elapsed_time = time.time() - tiempo_comienzo
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            uso_cpu.append(cpu_usage)
            uso_memoria.append(memory_usage)
            time_data.append(elapsed_time)

            ruta_imagen = os.path.join(directorio_raiz, imagen_nombre)

            img = Image.open(ruta_imagen)

            # VECTORIZAR
            cara = mtcnn(img)
            if cara is not None:
                embedding_cara = encoder.forward(cara.reshape((1, 3, 160, 160))).detach().cpu()
                vectors.append(embedding_cara)
                nombre_archivo_vector = f'vector_{contador}.txt'
                ruta_guardado_vector = os.path.join(output_directory_vectors, nombre_archivo_vector)

                with open(ruta_guardado_vector, 'w') as file:
                    for valor in embedding_cara.numpy().flatten():
                        file.write(f'{valor} ')
                contador += 1
                
        except ValueError as e:
            print(f'Error en la imagen {imagen_nombre}: {str(e)}')
            
            ruta_imagen = os.path.join(directorio_raiz, imagen_nombre)
            os.remove(ruta_imagen)

            continue

        except Exception as e:
            print(f'Error en la imagen {imagen_nombre}: {str(e)}')
            continue 

    for i in range(len(vectors)):
        elapsed_time = time.time() - tiempo_comienzo
        cpu_usage = get_cpu_usage()
        memory_usage = get_memory_usage()

        uso_cpu.append(cpu_usage)
        uso_memoria.append(memory_usage)
        time_data.append(elapsed_time)

        for j in range(i + 1, len(vectors)):
            distancia_euclidiana = np.linalg.norm(vectors[i] - vectors[j])
            print(f'Distancia entre vector {i+1} y vector {j+1}: {distancia_euclidiana}')

    print(f'Total de imágenes procesadas: {len(vectors)}')
    print(f'Tiempo total de ejecución: {time.time() - tiempo_comienzo} segundos')
    print(f'Memoria utilizada: {get_memory_usage()} MB')
    return uso_cpu, uso_memoria, time_data


if __name__ == "__main__":

    archivo_configuracion = "config.yaml"
    with open(archivo_configuracion, "r") as archivo:
        configuracion = yaml.safe_load(archivo)

    directorio_raiz = configuracion["configuracion"]["input_path"]
    output_directory_vectors = configuracion["configuracion"]["output_vectors"]
    ruta_plot = configuracion["configuracion"]["output_graph"]

    cpu_usage_data = []
    memory_usage_data = []
    time_data = []
    start_time = time.time()
    print("Comienzo\n")
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        cpu_usage_data, memory_usage_data, time_data = main_process(start_time)
        print(f'Tiempo total de ejecución: {time.time() - start_time} segundos')

    except KeyboardInterrupt:
        pass

    finally:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.plot(time_data, cpu_usage_data, label='Uso de CPU')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Uso de CPU (%)')
        plt.title('Rendimiento de la CPU durante la ejecución del proceso principal')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(time_data, memory_usage_data, label='Uso de Memoria')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Uso de Memoria (MB)')
        plt.title('Rendimiento de la Memoria durante la ejecución del proceso principal')
        plt.legend()

        plt.tight_layout()
        plt.savefig(ruta_plot)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(20)
