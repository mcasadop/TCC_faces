import time
import psutil
import os
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from mpi4py import MPI
import cProfile
import pstats

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)  # Intervalo de muestreo de 1 segundo

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Uso de memoria en MB

def main_process(start_time, rank):
    archivos = [archivo for archivo in os.listdir(directorio_raiz)]
    contador = 1
    uso_cpu = []
    uso_memoria = []
    time_data = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device(process.py, rank {rank}): {device}')
    mtcnn = MTCNN(
        select_largest=True,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        image_size=160,
        device=device
    )
    encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()

    vectors = []

    for imagen_nombre in archivos:
        print(f'Procesando: {imagen_nombre}')
        try:
            elapsed_time = time.time() - start_time
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            uso_cpu.append(cpu_usage)
            uso_memoria.append(memory_usage)
            time_data.append(elapsed_time)

            ruta_imagen = os.path.join(directorio_raiz, imagen_nombre)
            img = Image.open(ruta_imagen)

            cara = mtcnn(img)
            if cara is not None:
                embedding_cara = encoder.forward(cara.reshape((1, 3, 160, 160))).detach().cpu()
                vectors.append(embedding_cara)

                for i in range(len(vectors) - 1):
                    distancia_euclidiana = np.linalg.norm(vectors[i] - embedding_cara)
                    print(f'Distancia entre vector {i+1} y vector {contador}: {distancia_euclidiana}')

                nombre_archivo_vector = f'vector_{contador}.txt'
                ruta_guardado_vector = os.path.join(output_directory_vectors, nombre_archivo_vector)

                with open(ruta_guardado_vector, 'w') as file:
                    for valor in embedding_cara.numpy().flatten():
                        file.write(f'{valor} ')

                contador += 1

        except Exception as e:
            print(f'Error en la imagen {imagen_nombre}: {str(e)}')
            continue

    print(f'Total de imágenes procesadas por proceso {rank}: {len(vectors)}')
    print(f'Tiempo total de ejecución del proceso {rank}: {time.time() - start_time} segundos')
    print(f'Memoria utilizada por proceso {rank}: {get_memory_usage()} MB')
    return uso_cpu, uso_memoria, time_data

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    directorio_raiz = r'C:\Users\Eduardo\Downloads\PortableGit\TCC_faces\Segunda\Input_Prueba'
    output_directory_vectors = r'C:\Users\Eduardo\Downloads\PortableGit\TCC_faces\Segunda\Output_Prueba\Vectors'

    cpu_usage_data = []
    memory_usage_data = []
    time_data = []
    start_time = time.time()
    print("Comienzo\n")
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        comm.Barrier()
        if rank == 0:
            archivos = [archivo for archivo in os.listdir(directorio_raiz)]
            archivos_por_proceso = [archivos[i::size] for i in range(size)]
        else:
            archivos_por_proceso = None

        archivos_por_proceso = comm.scatter(archivos_por_proceso, root=0)
        cpu_usage_data, memory_usage_data, time_data = main_process(start_time, rank)
        cpu_usage_data = comm.gather(cpu_usage_data, root=0)
        memory_usage_data = comm.gather(memory_usage_data, root=0)
        time_data = comm.gather(time_data, root=0)
            
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
        plt.savefig(r'C:\Users\Eduardo\Downloads\PortableGit\TCC_faces\Segunda\Output_Prueba\performance_parallel1.png')

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(20)