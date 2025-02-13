import subprocess
import os

def clone_huggingface_repo(repo_url, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    try:
        result = subprocess.run(["git", "clone", repo_url, destination_dir], 
                                check=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)     
        print("Git clone executed successfully.")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cloning the repository: {e.stderr.decode()}")