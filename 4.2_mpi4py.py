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

def procesar_segmento(segmento, kernel_45):
    """
    Procesa un segmento de la imagen utilizando el filtro.
    """
    return detectar_diagonales_45(segmento, kernel_45)

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
    plt.title("Filtrado MPI")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500, :500], cmap='binary_r', aspect='auto')
    plt.savefig("ResultadosFiltrados/RFiltradoMPI.png")

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configuration for loading .dat file
    input_filename = 'ArchivosDAT/dotplot_result_thread.dat'

    # Root process reads sequences and prepares input matrix
    if rank == 0:
        # Read sequences
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

        # Load the .dat file as a grayscale image
        input_map = np.memmap(input_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
        imagen = 255 - (input_map.astype(np.uint8) * 255)
    else:
        # Other processes initialize these variables
        imagen = None
        seq1_len = None
        seq2_len = None

    # Broadcast sequence lengths and prepare image segments
    seq1_len = comm.bcast(seq1_len, root=0)
    seq2_len = comm.bcast(seq2_len, root=0)

    # Distribute image segments across processes
    if rank == 0:
        # Split image into segments
        segmentos = np.array_split(imagen, size, axis=0)
    else:
        segmentos = None

    # Scatter image segments to all processes
    local_segment = comm.scatter(segmentos, root=0)

    # Kernel configuration
    kernel_45 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Process local segment
    inicio = datetime.now() if rank == 0 else None
    local_resultado = procesar_segmento(local_segment, kernel_45)

    # Gather results from all processes
    resultados = comm.gather(local_resultado, root=0)

    # Final processing on root process
    if rank == 0:
        fin = datetime.now()
        
        # Combine results
        diagonales_45 = np.vstack(resultados)

        # Calculate and print processing time
        delta = fin - inicio
        print("Inicio:", inicio)
        print("Fin:", fin)
        print("Duración:", delta)

        # Save the result matrix to a .dat file
        output_filename = 'ArchivosDAT/Filtrados/diagonales_45_mpi.dat'
        output_map = np.memmap(output_filename, dtype='uint8', mode='w+', shape=diagonales_45.shape)
        output_map[:] = diagonales_45[:]
        output_map.flush()

        # Generate dot plot
        plot_dotplot(output_filename, seq1_len, seq2_len)

        print(f"Procesamiento completado. Resultado guardado en {output_filename}.")

if __name__ == '__main__':
    main()
