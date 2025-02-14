import subprocess
import os

from os import path
from setuptools import setup, find_packages
from setuptools.command.install import install

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

here = path.abspath(path.dirname(__file__))

class CustomInstallCommand(install):
    """Customized install command to download models."""
    def run(self) -> None:
        repo_url = "https://huggingface.co/xqh/BCR-V-BERT"
        destination_dir = path.join(here, "BCR_V_BERT/model_pretrained")
        
        print("Starting to download the model repository from Hugging Face...")
        try:
            clone_huggingface_repo(repo_url, destination_dir)
            print("Model repository downloaded successfully.")
        except Exception as e:
            print(f"Failed to download the model repository: {e}")
        super().run()

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

requirements_file = path.join(here, 'requirements.txt')
long_description_file = path.join(here, 'README.md')

try:
    all_reqs = read_file(requirements_file).split('\n')
    long_description = read_file(long_description_file)
except FileNotFoundError as e:
    print(f"Error reading file: {e}")
    all_reqs = []
    long_description = ""

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='BCR_V_BERT',  
    version='1.0.0',  
    author='Qihang Xu, Jian Zhang',  
    description='BCR_V_BERT, an antibody-specific transformer language model pre-trained separately on 372M antibody heavy sequences and 3M antibody light sequences. The model can embed the CDR3 or CDR sequences of light and heavy chain antibodies for various downstream tasks.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    author_email='jian_zhang@tju.edu.cn',  
    url='https://github.com/ZhangLabTJU/BCR-V-BERT',  
    license='CC BY-NC-SA 4.0',
    include_package_data=True, 
    packages=find_packages(),  
    cmdclass={
        'install': CustomInstallCommand,
    },
    install_requires=install_requires, 
    classifiers=[
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.9"
)
