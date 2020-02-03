"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To install this package, open anaconda command prompt and 
# navigate to the directory containing setup.py.
#    * For Ethan - "python setup.py install --develop"
#    * Anyone else - "python setup.py install --user"
# in command line

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