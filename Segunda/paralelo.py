import os
import time
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import psutil
import matplotlib.pyplot as plt
from multiprocessing import Manager, Process
import yaml

def get_cpu_usage():
    return psutil.cpu_percent(interval=0.1)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def monitor_resource_usage(usage_data, lock, interval=1):
    try:
        while True:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().used / (1024 ** 2)

            with lock:
                usage_data['cpu'].append(cpu)
                usage_data['memory'].append(memory)
                usage_data['time'].append(time.time() - usage_data['start_time'])
            
            print(f"Recopilado: CPU: {cpu}%, Memoria: {memory}MB")
            time.sleep(interval)
    except Exception as e:
        print(f"Error en el proceso de monitoreo: {e}")

def vectorize_images(images, lock, usage_data):
    vectors = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        select_largest=True,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        image_size=160,
        device=device
    )
    encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()

    for ruta_imagen in images:
        try:
            img = Image.open(ruta_imagen)

            cara = mtcnn(img)
            if cara is not None:
                with lock:
                    usage_data['cpu'].append(get_cpu_usage())
                    usage_data['memory'].append(get_memory_usage())
                    usage_data['time'].append(time.time() - usage_data['start_time'])
                
                embedding_cara = encoder.forward(cara.reshape((1, 3, 160, 160))).detach().cpu()
                vectors.append(embedding_cara)

        except ValueError as e:
            print(f'Error en la imagen {ruta_imagen}: {str(e)}')

    return vectors

def calculate_euclidean_distances(vectors):
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distancia_euclidiana = np.linalg.norm(vectors[i] - vectors[j])
            print(f'Distancia entre vector {i+1} y vector {j+1}: {distancia_euclidiana}')

def process_images_batch(batch, lock, usage_data):
    start_time = time.time()
    vectors_batch = vectorize_images(batch, lock, usage_data)
    elapsed_time = time.time() - start_time

    with lock:
        usage_data['cpu'].append(get_cpu_usage())
        usage_data['memory'].append(get_memory_usage())
        usage_data['time'].append(time.time() - usage_data['start_time'])

    return vectors_batch

def main():
    manager = Manager()
    usage_data = manager.dict(cpu=[], memory=[], time=[], start_time=time.time())
    lock = manager.Lock()

    monitor_process = Process(target=monitor_resource_usage, args=(usage_data, lock))
    monitor_process.daemon = True
    monitor_process.start()

    archivo_configuracion = r"D:\Users\Miguel\Documents\TCC_faces\Segunda\config.yaml"
    with open(archivo_configuracion, "r") as archivo:
        configuracion = yaml.safe_load(archivo)

    directorio_raiz = configuracion["configuracion"]["input_path"]
    output_directory_vectors = configuracion["configuracion"]["output_vectors"]

    archivos = [os.path.join(directorio_raiz, archivo) for archivo in os.listdir(directorio_raiz)]

    batch_size = 50
    batches = [archivos[i:i+batch_size] for i in range(0, len(archivos), batch_size)]

    num_cores = os.cpu_count()
    num_jobs = len(batches)

    print(f'Número de cores disponibles: {num_cores}')
    print(f'Cantidad total de trabajos a ejecutar simultáneamente: {num_jobs}')

    tiempo_ini = time.time()
    print('Comienza el multiproceso para la vectorización.')
    with ProcessPoolExecutor() as executor:
        all_vectors = list(executor.map(process_images_batch, batches, [lock]*len(batches), [usage_data]*len(batches)))

    print('Fin del multiproceso')
    vectors = [vector for vectors_batch in all_vectors for vector in vectors_batch]

    calculate_euclidean_distances(vectors)

    monitor_process.terminate()
    monitor_process.join()

    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_ini
    uso_medio_cpu = np.mean(usage_data['cpu'])
    uso_medio_memoria = np.mean(usage_data['memory'])

    print(f'Tiempo total de ejecución: {tiempo_total} segundos')
    print(f'Uso medio de CPU: {uso_medio_cpu}%')
    print(f'Uso medio de memoria: {uso_medio_memoria} MB')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(usage_data['time'], usage_data['cpu'], label='Uso de CPU (%)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Uso de CPU (%)')
    plt.title('Uso de CPU a lo largo del tiempo')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(usage_data['time'], usage_data['memory'], label='Uso de Memoria (MB)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Uso de Memoria (MB)')
    plt.title('Uso de Memoria a lo largo del tiempo')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()