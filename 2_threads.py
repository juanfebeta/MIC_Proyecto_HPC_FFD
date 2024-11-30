import threading
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os

def worker(seq1_len, seq1_chunk, seq2, start_idx, result_filename, lock):
    """
    Worker function for threading-based dotplot calculation
    
    Args:
    - seq1_len: Total length of first sequence
    - seq1_chunk: Chunk of first sequence to process
    - seq2: Second sequence
    - start_idx: Starting index of the chunk in the original sequence
    - result_filename: Filename for memory-mapped result
    - lock: Threading lock to ensure thread-safe file writing
    """
    # Calculate the outer equality
    result_chunk = np.equal.outer(seq1_chunk, seq2)

    # Use lock to ensure thread-safe writing to memory-mapped file
    with lock:
        # Memory-mapped file where results are stored
        result_map = np.memmap(result_filename, dtype='bool', mode='r+', shape=(seq1_len, len(seq2)))

        # Write the computed chunk to the correct location
        result_map[start_idx:start_idx + len(seq1_chunk), :] = result_chunk

def plot_dotplot(result_filename, seq1_len, seq2_len):
    """
    Plot the dot plot from the memory-mapped result.
    
    Args:
    - result_filename: Filename of memory-mapped result
    - seq1_len: Length of first sequence
    - seq2_len: Length of second sequence
    """
    result_map = np.memmap(result_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
    plt.figure(figsize=(10,10))
    plt.title("Dotplot Threading")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500,:500], cmap='binary', aspect='auto')
    plt.savefig("./Resultados/ResultadoTHR_TPB.png")

def main():
    begin = time.time()
    print(datetime.today())

    # Read first sequence
    with open('../archivos_dotplot/elemento1.fasta', 'r') as file:
        seq1 = file.read()

    # Remove the first line and line breaks
    seq1 = ''.join(seq1.split('\n')[1:])

    # Read second sequence
    with open('../archivos_dotplot/elemento2.fasta', 'r') as file:
        seq2 = file.read()

    # Remove the first line and line breaks
    seq2 = ''.join(seq2.split('\n')[1:])

    # Convert sequences to numpy arrays
    seq1 = np.array(list(seq1)).astype('str')
    seq2 = np.array(list(seq2)).astype('str')

    # Mapping for nucleotide encoding
    mapping = {'A': 0, 'C': 1, 'G': 2, 'N': 3, 'T': 4}
    seq1 = np.vectorize(mapping.get)(seq1).astype('uint8')
    seq2 = np.vectorize(mapping.get)(seq2).astype('uint8')

    # Get lengths of sequences
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Number of threads
    num_threads = 10

    # Create a memory-mapped file to store the result
    result_filename = './dotplot_result_thread.dat'
    result_map = np.memmap(result_filename, dtype='bool', mode='w+', shape=(seq1_len, seq2_len))

    # Create a lock for thread-safe file writing
    file_lock = threading.Lock()

    # Split seq1 into chunks
    chunk_size = seq1_len // num_threads
    threads = []

    # Create threads to compute outer equality
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i != num_threads - 1 else seq1_len

        seq1_chunk = seq1[start_idx:end_idx]
        
        # Create a thread for each chunk
        t = threading.Thread(target=worker, args=(seq1_len, seq1_chunk, seq2, start_idx, result_filename, file_lock))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

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

if __name__ == '__main__':
    main()