import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from datetime import datetime


def detectar_diagonales_45(imagen, kernel_45):
    """
    Aplica un filtro de 45 grados a la imagen y normaliza el resultado.
    """
    diagonal_45 = cv2.filter2D(imagen, -1, kernel_45)
    diagonal_45_norm = cv2.normalize(diagonal_45, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return diagonal_45_norm


def procesar_segmento(segmento, kernel_45, resultado, idx):
    """
    Procesa un segmento de la imagen utilizando el filtro y guarda el resultado.
    """
    resultado[idx] = detectar_diagonales_45(segmento, kernel_45)


def plot_dotplot(result_filename, seq1_len, seq2_len):
    """
    Plot the dot plot from the memory-mapped result.

    Args:
    - result_filename: Filename of memory-mapped result
    - seq1_len: Length of first sequence
    - seq2_len: Length of second sequence
    """
    result_map = np.memmap(result_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
    plt.figure(figsize=(10, 10))
    plt.title("Filtrado Threading")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500, :500], cmap='binary_r', aspect='auto')
    plt.savefig("./ResultadosFiltrados/RFiltradoTHR.png")

tig = datetime.now()
print('iniciando')

# Configuración para cargar el archivo .dat
input_filename = './ArchivosDAT/dotplot_result_thread.dat'

# Leer secuencias y preparar la matriz de entrada
with open('archivos_dotplot/elemento1.fasta', 'r') as file:
    seq1 = file.read().splitlines()[1:]

with open('archivos_dotplot/elemento2.fasta', 'r') as file:
    seq2 = file.read().splitlines()[1:]

seq1 = np.array(list("".join(seq1))).astype('str')
seq2 = np.array(list("".join(seq2))).astype('str')

mapping = {'A': 0, 'C': 1, 'G': 2, 'N': 3, 'T': 4}
seq1 = np.vectorize(mapping.get)(seq1).astype('uint8')
seq2 = np.vectorize(mapping.get)(seq2).astype('uint8')

seq1_len = len(seq1)
seq2_len = len(seq2)

# Cargar el archivo .dat como una imagen en escala de grises
input_map = np.memmap(input_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
imagen = 255 - (input_map.astype(np.uint8) * 255)  # Convertir de booleano a escala de grises (0 y 255)

# Configuración del kernel de filtro
kernel_45 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float32)

# Parametrizar la cantidad de hilos
num_hilos = 10  # Cambiar este valor para ajustar la cantidad de hilos

# Dividir la imagen en segmentos horizontales según el número de hilos
segmentos = np.array_split(imagen, num_hilos, axis=0)

# Procesar cada segmento en un hilo
resultados = [None] * num_hilos
hilos = []

inicio = datetime.now()
print("Tiempo de inicio threads:", inicio)

for idx, segmento in enumerate(segmentos):
    print("longitud ",segmento.shape)
    hilo = threading.Thread(target=procesar_segmento, args=(segmento, kernel_45, resultados, idx))
    hilos.append(hilo)
    hilo.start()

# Esperar a que todos los hilos terminen
for hilo in hilos:
    hilo.join()

fin = datetime.now()
print("Tiempo de fin threads:", fin)

# Combinar los resultados
diagonales_45 = np.vstack(resultados)

# Calcular el delta
delta = fin - inicio
print("Duración threads:", delta)

# Guardar la matriz resultante en un archivo .dat
output_filename = 'ArchivosDat/Filtrados/diagonales_threads_45.dat'
output_map = np.memmap(output_filename, dtype='uint8', mode='w+', shape=diagonales_45.shape)
output_map[:] = diagonales_45[:]
output_map.flush()

# Generar el dotplot
plot_dotplot(output_filename, seq1_len, seq2_len)

print(f"Procesamiento completado. Resultado guardado en {output_filename}.")

delta2 = datetime.now() - tig
print("Duración total:", delta2)