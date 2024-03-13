from setuptools import setup, find_packages

setup(
    name="synthetic_module",
    version="0.1.0",
    description="package for synthetic diagram generation",
    author="Syrine Kalleli",
    author_email="cyrine.kalleli@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
)
