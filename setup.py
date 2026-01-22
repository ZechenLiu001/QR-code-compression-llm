from setuptools import setup, find_packages

setup(
    name="image-context-compression",
    version="0.1.0",
    description="Image-based long context compression experiment framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "accelerate>=0.25.0",
        "qwen-vl-utils>=0.0.8",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "qrcode>=7.4.0",
        "pylibdmtx>=0.1.10",
        "pyzbar>=0.1.9",
        "pyyaml>=6.0",
        "jsonlines>=4.0.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.41.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
    ],
)
