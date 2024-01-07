import subprocess
import matplotlib.pyplot as plt

# Ejecutar process.py
subprocess.run(['python', 'process.py'])

# Ejecutar process_M1.py
subprocess.run(['python', 'process_M1.py'])

# Mostrar el gráfico
plt.show()