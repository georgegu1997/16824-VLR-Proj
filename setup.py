#!/usr/bin/env python
from distutils.core import setup, Extension
from setuptools import find_packages

setup(
    name='ppf_net',
    version='0.1dev',
    author='Qiao Gu, Yanjia Duan',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    description= 'PyTorch code for 16-824 course project',
    long_description='',
)

