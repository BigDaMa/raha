from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='raha',
    version='1.1',
    packages=[''],
    url='https://github.com/BigDaMa/raha/',
    license='Apache 2.0',
    author='',
    author_email='',
    description='Raha: A Configuration-Free Error Detection System',
    install_requires=required
)
