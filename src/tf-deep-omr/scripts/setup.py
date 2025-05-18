from setuptools import setup, find_packages

setup(
    name="tf-deep-omr",
    version="0.1.0",
    description="Deep Optical Music Recognition using TensorFlow",
    author="SuperSMM Team",
    author_email="info@supersmm.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.19.0",
        "opencv-python>=4.0.0",
        "pyyaml>=5.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "tf-deep-omr-train=train:main",
        ],
    },
)
