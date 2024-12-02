# PROYECTO HPC CÁLCULO EN PARALELO PARA GENERAR ‘DOTPLOT’ DE SECUENCIAS GENÉTICAS 


## Instrucciones de configuración

### 1. Cree un entorno de Python

Para crear un nuevo entorno virtual de Python, ejecute el siguiente comando:

```bash
python -m venv env
```

### 2. Active el entorno

Active el entorno virtual con el siguiente comando:

- En Windows:
```bash
.\env\Scripts\activate
```
- En macOS y Linux:
```bash
source env/bin/activate
```

### 3. Instale las bibliotecas necesarias

Instale las bibliotecas necesarias especificadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configuración adicional

Para modificar los datos de entrada, dirijase a las lineas correspondientes, relacionadas como 'InputFiles'

## Uso

En particular para la ejecucion de los archivos .py relacionados con MPI4PY, ejecute la siguiente linea de comandos:

mpiexec -n 10 python 4.1_mpi4py.py

mpiexec -n 10 python 4.2_mpi4py.py