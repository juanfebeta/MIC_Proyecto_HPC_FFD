#mpiexec -n 10 python 4_mpi4py.py

from mpi4py import MPI
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys

def worker(comm):
    """
    Worker process logic
    """
    # Receive data from root
    seq1_len, seq1_chunk, seq2, start_idx, result_filename = comm.recv(source=0, tag=11)

    # Calculate the outer equality
    result_chunk = np.equal.outer(seq1_chunk, seq2)

    # Memory-mapped file where results are stored
    result_map = np.memmap(result_filename, dtype='bool', mode='r+', shape=(seq1_len, len(seq2)))

    # Write the computed chunk to the correct location
    result_map[start_idx:start_idx + len(seq1_chunk), :] = result_chunk

    # Send completion signal back to root
    comm.send(True, dest=0, tag=12)

def plot_dotplot(result_filename, seq1_len, seq2_len):
    """
    Plot the dot plot from the memory-mapped result.
    """
    result_map = np.memmap(result_filename, dtype='bool', mode='r', shape=(seq1_len, seq2_len))
    plt.figure(figsize=(10,10))
    plt.title("Dotplot MPI")
    plt.xlabel("Seq2")
    plt.ylabel("Seq1")
    plt.imshow(result_map[:500,:500], cmap='binary', aspect='auto')
    plt.savefig("./Resultados/ResultadoMPI.png")

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Root process (rank 0) handles sequence loading and distribution
    if rank == 0:
        begin = time.time()
        print(f"Starting MPI Dotplot (Total processes: {size})")

        # Read first sequence
        with open('./archivos_dotplot/elemento1.fasta', 'r') as file:
            seq1 = file.read()

        # Remove the first line and line breaks
        seq1 = ''.join(seq1.split('\n')[1:])

        # Read second sequence
        with open('./archivos_dotplot/elemento2.fasta', 'r') as file:
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

        # Create a memory-mapped file to store the result
        result_filename = './dotplot_result_mpi.dat'
        result_map = np.memmap(result_filename, dtype='bool', mode='w+', shape=(seq1_len, seq2_len))

        # Calculate chunk sizes for distribution
        # Ensure we have enough processes
        if size < 2:
            print("Need at least 2 processes. Exiting.")
            sys.exit(1)

        # Adjust for 10 cores (9 workers + 1 root)
        num_workers = 9
        chunk_size = seq1_len // num_workers

        # Distribute sequence chunks to worker processes
        for i in range(1, 10):  # Explicitly use 1-9 as worker ranks
            start_idx = (i-1) * chunk_size
            end_idx = start_idx + chunk_size if i != 9 else seq1_len
            
            seq1_chunk = seq1[start_idx:end_idx]
            
            # Send chunk information to each worker
            comm.send((seq1_len, seq1_chunk, seq2, start_idx, result_filename), dest=i, tag=11)

        # Root process waits for all workers to complete
        for i in range(1, 10):  # Explicitly wait for ranks 1-9
            comm.recv(source=i, tag=12)

        end = time.time()
        print(f"MPI Computation time: {end-begin} seconds")

        # Plot the result
        plot_dotplot(result_filename, seq1_len, seq2_len)

        # Cleanup (close the memory-mapped file)
        del result_map
        
        total_end = time.time()
        print(f"Total execution time: {total_end-begin} seconds")

    else:
        # Worker processes
        worker(comm)

if __name__ == '__main__':
    main()