from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

with open(path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='BCR_V_BERT',
    version='1.0.0',
    author='ZJ-lab',
    description='BCR_V_BERT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email='jian_zhang@tju.edu.cn',
    url='https://github.com/ZhangLabTJU/BCR-V-BERT',
    license='MIT',
    include_package_data=True, 
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.9"
)
