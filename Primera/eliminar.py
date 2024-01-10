import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

def eliminar_imagen(ruta_imagen):
    try:
        os.remove(ruta_imagen)
        print(f'Imagen eliminada: {ruta_imagen}')
    except Exception as e:
        print(f'Error al eliminar la imagen {ruta_imagen}: {str(e)}')

def procesar_imagenes(directorio_raiz):
    archivos = [archivo for archivo in os.listdir(directorio_raiz)]
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

    for imagen_nombre in archivos:
        try:
            # Construir la ruta completa de la imagen
            ruta_imagen = os.path.join(directorio_raiz, imagen_nombre)

            # Abrir la imagen
            img = Image.open(ruta_imagen)

            # VECTORIZAR
            cara = mtcnn(img)
            if cara is not None:  # Verificar si se detectó una cara
                # Realizar el embedding (si es necesario)
                embedding_cara = encoder.forward(cara.reshape((1, 3, 160, 160))).detach().cpu()

        except Exception as e:
            print(f'Error en la imagen {imagen_nombre}: {str(e)}')
            eliminar_imagen(ruta_imagen)
            continue  # Continuar con la siguiente imagen en caso de error

if __name__ == "__main__":
    directorio_raiz = r'D:\Users\Miguel\Documents\TCC_faces\Primera\input\Humans'
    procesar_imagenes(directorio_raiz)
