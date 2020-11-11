from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="raha",
    version="1.2",
    packages=find_packages(),
    url="https://github.com/bigdama/raha",
    license="Apache 2.0",
    author="Mohammad Mahdavi",
    author_email="moh.mahdavi.l@gmail.com",
    description="Raha and Baran: An End-to-End Data Cleaning System",
    keywords=["Data Cleaning", "Machine Learning", "Error Detection", "Error Correction", "Data Repairing"],
    install_requires=required,
    include_package_data=True,
    download_url="https://github.com/BigDaMa/raha/archive/v1.2.tar.gz"
)
