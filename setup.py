#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setup(name='FCD',
      version='1.0',
      author='',
      author_email='',
      description='Fréchet ChEMNet Distance',
      url='https://github.com/bioinf-jku/FCD',
      packages=['fcd'],
      license='LGPLv3',
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=[
          'keras',
          'numpy',
          'scipy',
          'tensorflow'
      ],
      extras_require={
          'rdkit': ['rdkit'],
      },
      include_package_data=True,
      )
