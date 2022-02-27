import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ocotillo",
    packages=["."],
    version="1.0.2",
    author="James",
    author_email="james@adamant.ai",
    description="A simple & fast speech transcription toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neonbjb/ocotillo",
    project_urls={},
    install_requires=[
        'tqdm',
        'scipy',
        'torch>=1.8',
        'torchaudio>0.9',
        'audio2numpy',
        'transformers',
        'tokenizers',
        'requests',
        'ffmpeg',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    download_url = 'https://github.com/neonbjb/ocotillo/archive/refs/tags/1.0.2.tar.gz',
    python_requires=">=3.6",
)