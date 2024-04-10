import os
import sys
import shutil
from sklearn.model_selection import train_test_split

def move_images(source_dir, dest_dir, test_size=0.2, random_state=None):
    # Lista todos os arquivos na pasta de origem
    image_files = os.listdir(source_dir)
    # Divide os arquivos em conjuntos de treinamento e teste
    _, test_images, _, _ = train_test_split(image_files, image_files, test_size=test_size, random_state=random_state)
    
    # Cria a pasta de destino se ela não existir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Move os arquivos de teste para a pasta de destino
    for image in test_images:
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(dest_dir, image)
        shutil.move(source_path, dest_path)
        print(f"Moved {image} to {dest_dir}")

if __name__ == "__main__":
    # Verifica se o número correto de argumentos foi fornecido
    if len(sys.argv) != 3:
        print("Uso: python Create_test_list.py --origem=<caminho_para_pasta_de_origem> --destino=<caminho_para_pasta_de_destino>")
        sys.exit(1)

    # Extrai os caminhos das pastas de origem e destino dos argumentos
    args = sys.argv[1:]
    source_directory = None
    destination_directory = None
    for arg in args:
        if arg.startswith("--origem="):
            source_directory = arg.split("=")[1]
        elif arg.startswith("--destino="):
            destination_directory = arg.split("=")[1]

    if source_directory is None or destination_directory is None:
        print("Ambos os caminhos de origem e destino devem ser fornecidos.")
        sys.exit(1)

    # Chama a função para mover as imagens
    move_images(source_directory, destination_directory)
