from setuptools import find_packages, setup

setup(
    name='LiquidBayes',
    version='0.2',
    description='LiquidBayes is a Bayesian Network (BN) for inferring clonal prevalences from liquid biopsy sequencing data',
    author='Kevin Yang',
    author_email='kevinyang10@gmail.com',
    url='https://github.com/Roth-Lab/LiquidBayes',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'liquid-bayes=src.cli:run',
        ]
    }
)
