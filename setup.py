from setuptools import setup, find_packages

setup(
    name='adaptive-codegen',
    version='0.1.0',
    description='Adaptive embedded convex optimization code generator',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=['numpy>=1.24'],
    extras_require={
        'plots': ['matplotlib>=3.7'],
        'ref':   ['cvxpy>=1.4', 'scipy>=1.11'],
    },
)
