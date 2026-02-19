"""
STF-Mamba V8.0 -- Package Setup
Install locally: pip install -e .
Then: import stf_mamba
"""

from setuptools import setup, find_packages

setup(
    name="stf_mamba",
    version="8.0.0",
    author="Abdel Rahman Madboly",
    description="STF-Mamba V8.0: Semantic Temporal Forensics via Hydra-Mamba and DINOv2",
    url="https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "einops>=0.7.0",
    ],
)
