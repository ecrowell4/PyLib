"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name='nlopy',
    version='0.1.0',
    description='A Python package for nonlinear optics',
    url='',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'mpmath'],
)