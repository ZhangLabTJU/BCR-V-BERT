from os import path
from setuptools import setup, find_packages
from setuptools.command.install import install
from BCR_V_BERT.utils.download_model import clone_huggingface_repo

here = path.abspath(path.dirname(__file__))
class CustomInstallCommand(install):
    """Customized install command to download models."""
    def run(self):
        install.run(self)
        repo_url = "https://huggingface.co/xqh/vbert"
        destination_dir = path.join(here, "BCR_V_BERT/model_pretrained")
        
        print("Starting to download the model repository...")
        clone_huggingface_repo(repo_url, destination_dir)
        print("Model repository downloaded successfully.")

with open(path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

with open(path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='BCR_V_BERT',  
    version='1.0.0',  
    author='ZJ-lab',  
    description='BCR_V_BERT, an antibody-specific transformer language model pre-trained separately on 372M antibody heavy sequences and 3M antibody light sequences. The model can embed the CDR3 or CDR sequences of light and heavy chain antibodies', 
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
        'License :: OSI Approved :: CC BY-NC-SA 4.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.9"
)
