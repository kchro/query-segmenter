"""Build script for setuptools."""
import setuptools

with open("README.md", "r") as fh:
    README = fh.read()

setuptools.setup(
    name="query_segmenter",
    version="0.0.1",
    author="kchro",
    description="A Query Segmenter Toolkit",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/kchro/query-segmenter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
