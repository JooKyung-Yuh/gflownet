"""
Setup script for Logic Gate Network GFlowNet
"""
from setuptools import setup, find_packages

setup(
    name="lgn-gflownet",
    version="0.1.0",
    description="Logic Gate Network Generation with GFlowNet",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
    author="GFlowNet Research",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
