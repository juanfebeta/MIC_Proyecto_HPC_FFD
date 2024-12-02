import cv2
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime
import os


def detectar_diagonales_45(imagen, kernel_45):
    """
    Aplica un filtro de 45 grados a la imagen y normaliza el resultado.
    """
    diagonal_45 = cv2.filter2D(imagen, -1, kernel_45)
    diagonal_45_norm = cv2.normalize(diagonal_45, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return diagonal_45_norm


def procesar_segmento(segmento, kernel_45, output_filename, start_row, total_rows):
    """
    Procesa un segmento de la imagen utilizando el filtro y guarda el resultado 
    directamente en un archivo memory-mapped.
    
    Args:
    - segmento: Parte de la imagen a procesar
    - kernel_45: Kernel para filtro diagonal
    - output_filename: Nombre del archivo de salida memory-mapped
    - start_row: Fila inicial de este segmento en la imagen completa
    - total_rows: Número total de filas en la imagen completa
    """
    # Abrir el archivo memory-mapped en modo lectura/escritura
    output_map = np.memmap(output_filename, dtype='uint8', mode='r+', 
                           shape=(total_rows, segmento.shape[1]))
    
    # Procesar el segmento
    resultado = detectar_diagonales_45(segmento, kernel_45)
    
    # Escribir el resultado en la posición correcta
    output_map[start_row:start_row+segmento.shape[0], :] = resultado
    
    # Asegurar que los cambios se guarden
    output_map.flush()


def plot_dotplot(result_filename, seq1_len, seq2_len):
    """
    Plot the dot plot from the memory-mapped result.

    Args:
    - result_filename: Filename of memory-mapped result
    - seq1_len: Length of first sequence
    - seq2_len: Length of second sequence
    """
    # Ensure the output directory exists
    os.makedirs('ResultadosFiltrados', exist_ok=True)
    
    result_map = np.memmap(result_filename, dtype='uint8', mode='r', 
                           shape=(seq1_len, seq2_len))
    plt.figure(figsize=(10, 10))
    plt.title("Filtrado Multiprocessing")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500, :500], cmap='binary_r', aspect='auto')
    plt.savefig("ResultadosFiltrados/RFiltradoMUL.png")
    plt.close()  # Close the plot to free up memory


def main():
    tgi = datetime.now()
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

    # Ensure the output directory exists
    os.makedirs('ArchivosDat/Filtrados', exist_ok=True)
    output_filename = 'ArchivosDat/Filtrados/diagonales_45_multiprocessing.dat'

    # Cargar el archivo .dat como una imagen en escala de grises
    input_map = np.memmap(input_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
    imagen = 255 - (input_map.astype(np.uint8) * 255)  # Convertir de booleano a escala de grises (0 y 255)

    # Configuración del kernel de filtro
    kernel_45 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Parametrizar la cantidad de procesos
    num_procesos = 5 

    # Crear el archivo memory-mapped de salida antes de dividir
    output_map = np.memmap(output_filename, dtype='uint8', mode='w+', 
                           shape=(seq1_len, seq2_len))
    del output_map  # Cerrar el mapeo para que los procesos puedan acceder

    # Dividir la imagen en segmentos horizontales según el número de procesos
    segmentos = np.array_split(imagen, num_procesos, axis=0)

    # Crear procesos
    procesos = []

    inicio = datetime.now()
    print("Tiempo de inicio:", inicio)

    # Calcular los índices de inicio de cada segmento
    start_rows = [sum(seg.shape[0] for seg in segmentos[:i]) for i in range(num_procesos)]

    # Crear y iniciar procesos
    for idx, (segmento, start_row) in enumerate(zip(segmentos, start_rows)):
        proceso = multiprocessing.Process(
            target=procesar_segmento, 
            args=(segmento, kernel_45, output_filename, start_row, seq1_len)
        )
        procesos.append(proceso)
        proceso.start()

    # Esperar a que todos los procesos terminen
    for proceso in procesos:
        proceso.join()

    fin = datetime.now()
    print("Tiempo de fin:", fin)

    # Calcular el delta
    delta = fin - inicio
    print("Duración Ejecucion codigo paralelo:", delta)

    # Generar el dotplot
    plot_dotplot(output_filename, seq1_len, seq2_len)

    fin = datetime.now()
    delta = fin - tgi
    print("Duración Ejecucion codigo total:", delta)

    print(f"Procesamiento completado. Resultado guardado en {output_filename}.")


if __name__ == '__main__':
    main()