from setuptools import setup, find_packages

setup(
    name="bsky-umap",
    version="0.1",
    packages=find_packages(),
    package_dir={'': 'umap'}  # This tells Python to look in the umap directory
)
