from setuptools import find_packages, setup

setup(
    name='LiquidBayes',
    version='1.0',
    description='Bayesian Model for monitoring cancer progression using liquid biopsies',
    author='Kevin Yang',
    author_email='kevinyang10@gmail.com',
    url='https://github.com/keviny2/LiquidBayes',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'liquid-bayes=liquid_bayes.cli:main',
        ]
    }
)
