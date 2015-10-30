import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sunkit-sst",
    version = "1.0",
    author = "Nabil Freij",
    description = ("A package for the manipulation of SST data."),
    license = "BSD",
    url = "https://github.com/nabobalis/sunkit-sst",
    packages=['sunkitsst'],
    install_requires=['numpy','sunpy'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
