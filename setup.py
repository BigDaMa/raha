from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="raha",
    version="1.1",
    packages=find_packages(),
    url="https://github.com/bigdama/raha",
    license="Apache 2.0",
    author="Mohammad Mahdavi",
    author_email="moh.mahdavi.l@gmail.com",
    description="Raha: A Configuration-Free Error Detection System",
    install_requires=required,
    include_package_data=True
)
