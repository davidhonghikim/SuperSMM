#!/usr/bin/env python3
"""
Setup script for the Deep OMR package.
This allows the package to be installed with pip.
"""

from setuptools import setup, find_packages

setup(
    name="deep_omr",
    version="0.1.0",
    description="Deep Optical Music Recognition using TensorFlow",
    author="Jorge Calvo-Zaragoza, David Rizo",
    author_email="",
    url="https://github.com/OMR-Research/tf-deep-omr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.16.0",
        "matplotlib>=3.0.0",
        "pyyaml>=5.1",
    ],
    entry_points={
        "console_scripts": [
            "deep-omr-train=deep_omr.cli.train:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
