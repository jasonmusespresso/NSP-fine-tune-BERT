"""
Fine-tune BERT
"""

from setuptools import setup

setup(
    name='fine-tuned-bert',
    version='0.1.0',
    packages=['NSP'],
    include_package_data=True,
    install_requires=[
        'torch==1.2.0',
        'torchvision==0.4.0',
        'pytorch-transformers',
        'tqdm',
    ],
)
