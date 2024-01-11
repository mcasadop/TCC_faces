# TCC_faces
Optimización de procesos de reconocimiento facial con visión artificial mediante técnicas secuenciales (/Primera) y técnicas de paralelismo (/Segunda).

## 
#### Authors: Miguel Casado Pina y Eduardo Marquina García
#### Contact: miguel.cpina@alumnos.upm.es || eduardo.marquina.garcia@alumnos.upm.es
#### Year: 2024

## Pre-Requisitos
El usuario debe tener el entorno Python instalado en su PC.

### Paso 1: 
Lo primero de todo se debe copiar el repositorio mediante el comando:  
> git clone https://github.com/mcasadop/TCC_faces/Primera

### Paso 2: 
Posicionarse en la carpeta en la que queremos tener nuestro proyecto e instalar las bibliotecas necesarias para su correcta ejecución: 
> pip install -r requierements.txt

## PRIMERA PARTE (OPTIMIZACIÓN SECUENCIAL)
### Instrucciones de Uso
Nuestro trabajo consiste en un script inicial (main_script1.py) y dos optimizaciones secuenciales sobre el mismo (main_script2.py y main_script3.py).
Para ejecutar los distintos script el usuario debe posicionarse en la carpeta /Primera y ejecutar:
> python3 ./main_script1
> 
> python3 ./main_script2
> 
> python3 ./main_script3

Al ejecutar cualquiera de estos scripts se generará en la carpeta /Ouput los gráficos respectivos al análisis del consumo mediante técnicas de Benchmarking.

## SEGUNDA PARTE (OPTIMIZACIÓN EN PARALELO)
### Instrucciones de Uso
Esta segunda parte solo consta de un script, para su ejecucion debemos ejecutar:
> python3 ./paralelo.py