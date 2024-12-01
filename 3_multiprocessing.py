import multiprocessing as mp
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def worker(seq1_len,seq1_chunk,seq2,start_idx, result_filename):

    # Calculate the outer equality
    result_chunk = np.equal.outer(seq1_chunk, seq2)

    # Memory-mapped file where results are stored
    result_map = np.memmap(result_filename, dtype='bool', mode='r+', shape=(seq1_len, len(seq2)))

    # Write the computed chunk to the correct location
    result_map[start_idx:start_idx + len(seq1_chunk), :] = result_chunk

# Plotting function
def plot_dotplot(result_filename, seq1_len, seq2_len):
    """
    Plot the dot plot from the memory-mapped result.
    """
    result_map = np.memmap(result_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
    plt.figure(figsize=(10,10))
    plt.title("Dotplot Multiprocessing")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500,:500], cmap='binary',aspect='auto')
    plt.savefig(f"./Resultados/ResultadoMUL.png")

if __name__ == '__main__':
    
    begin = time.time()
    print(datetime.today())

    with open('./archivos_dotplot/elemento1.fasta', 'r') as file:
        seq1 = file.read()

    # Remove the first line and line breaks
    seq1 = ''.join(seq1.split('\n')[1:])

    with open('./archivos_dotplot/elemento2.fasta', 'r') as file:
        seq2 = file.read()

    # Remove the first line and line breaks
    seq2 = ''.join(seq2.split('\n')[1:])

    seq1 = np.array(list(seq1)).astype('str')
    seq2 = np.array(list(seq2)).astype('str')

    mapping = {'A': 0, 'C': 1, 'G': 2,'N':3, 'T': 4}
    seq1 = np.vectorize(mapping.get)(seq1).astype('uint8')
    seq2 = np.vectorize(mapping.get)(seq2).astype('uint8')

    print(seq1)
    print(seq2)

    # Get lengths of sequences
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    #numero de procesadores
    num_chunks=10

    # Create a memory-mapped file to store the result
    result_filename = './dotplot_result.dat'
    result_map = np.memmap(result_filename, dtype='bool', mode='w+', shape=(seq1_len, seq2_len))

    # Split seq1 into chunks
    chunk_size = seq1_len // num_chunks
    processes = []

    # Create processes to compute outer equality
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i != num_chunks - 1 else seq1_len

        seq1_chunk = seq1[start_idx:end_idx]
        
        # Create a process for each chunk
        p = mp.Process(target=worker, args=(seq1_len,seq1_chunk, seq2, start_idx, result_filename))
        processes.append(p)
        p.start()

    # Ensure all processes finish
    for p in processes:
        p.join()

    end = time.time()
    print(datetime.today())
    print(f"Tiempo de ejecución paralela: {end-begin} segundos")

    # After parallel computation, plot the result
    plot_dotplot(result_filename, seq1_len, seq2_len)

    # Cleanup (close the memory-mapped file)
    del result_map
    
    end = time.time()
    print(datetime.today())
    print(f"Tiempo total de ejecución: {end-begin} segundos")