import cv2
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from datetime import datetime


def detectar_diagonales_45(imagen, kernel_45):
    """
    Aplica un filtro de 45 grados a la imagen y normaliza el resultado.
    """
    diagonal_45 = cv2.filter2D(imagen, -1, kernel_45)
    diagonal_45_norm = cv2.normalize(diagonal_45, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return diagonal_45_norm


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
    plt.title("Filtrado con MPI")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500, :500], cmap='binary_r', aspect='auto')
    plt.savefig("ResultadosFiltrados/RFiltradoMPI.png")


def main():
    # Tiempos de inicio del proceso principal
    start_time = datetime.now()

    # Inicialización de MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Cargar el archivo .dat y preparar las secuencias solo en el proceso principal
    input_filename = './ArchivosDAT/dotplot_result_thread.dat'

    if rank == 0:
        # Tiempo de inicio para la carga de datos
        load_start_time = datetime.now()

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

        # Dividir la imagen en segmentos horizontales según el número de procesos
        segmentos = np.array_split(imagen, size, axis=0)

        # Tiempo de finalización para la carga de datos
        load_end_time = datetime.now()
        print(f"Tiempos de carga: {load_end_time - load_start_time}")

    else:
        seq1_len = seq2_len = 0
        imagen = None
        kernel_45 = None
        segmentos = None

    # Enviar los datos necesarios a los otros procesos
    seq1_len = comm.bcast(seq1_len, root=0)
    seq2_len = comm.bcast(seq2_len, root=0)
    imagen = comm.bcast(imagen, root=0)
    kernel_45 = comm.bcast(kernel_45, root=0)
    segmentos = comm.scatter(segmentos, root=0)

    # Cada proceso filtra su segmento y mide el tiempo
    filter_start_time = datetime.now()
    resultado = detectar_diagonales_45(segmentos, kernel_45)
    filter_end_time = datetime.now()
    print(f"Tiempo de filtrado en proceso {rank}: {filter_end_time - filter_start_time}")

    # Recolectar los resultados de todos los procesos
    resultados = comm.gather(resultado, root=0)

    # El proceso principal combina los resultados
    if rank == 0:
        # Tiempo para combinar los resultados
        combine_start_time = datetime.now()
        diagonales_45 = np.vstack(resultados)

        # Guardar la matriz resultante en un archivo .dat
        output_filename = 'ArchivosDat/Filtrados/diagonales_45_mpi.dat'
        output_map = np.memmap(output_filename, dtype='uint8', mode='w+', shape=diagonales_45.shape)
        output_map[:] = diagonales_45[:]
        output_map.flush()

        # Generar el dotplot
        plot_dotplot(output_filename, seq1_len, seq2_len)

        # Tiempo de finalización para combinar los resultados
        combine_end_time = datetime.now()
        print(f"Tiempo de combinación y guardado: {combine_end_time - combine_start_time}")

        print(f"Procesamiento completado. Resultado guardado en {output_filename}.")

    # Tiempo total del proceso
    end_time = datetime.now()
    print(f"Tiempo total de ejecución: {end_time - start_time}")


if __name__ == '__main__':
    main()
