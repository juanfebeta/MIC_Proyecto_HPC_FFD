from mpi4py import MPI

def launch_mpi_processes():
    # Use mpiexec to launch this script with multiple processes
    comm = MPI.COMM_SELF.Spawn(
        'python', 
        args=['mul_mpi.py'], 
        maxprocs=10  # One root process, 10 worker processes
    )
    
    # Wait for the spawned processes to complete
    comm.Disconnect()

if __name__ == '__main__':
    launch_mpi_processes()