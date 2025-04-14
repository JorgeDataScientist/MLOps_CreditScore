import os

def print_tree(dir_path, prefix=""):
    """Imprime un árbol detallado de directorios y archivos."""
    # Obtener lista de elementos en el directorio
    contents = sorted(os.listdir(dir_path))
    # Ignorar ciertos directorios como venv o __pycache__ si quieres
    contents = [c for c in contents if c not in ['venv_credit_scoring', '__pycache__']]
    
    # Iterar sobre los elementos
    for index, item in enumerate(contents):
        path = os.path.join(dir_path, item)
        is_last = index == len(contents) - 1
        connector = "└── " if is_last else "├── "
        
        # Imprimir el elemento actual
        print(f"{prefix}{connector}{item}")
        
        # Si es un directorio, hacer recursión
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(path, new_prefix)

if __name__ == "__main__":
    # Ruta raíz del proyecto
    root_dir = os.getcwd()
    print(f"Árbol de directorios para: {root_dir}")
    print_tree(root_dir)