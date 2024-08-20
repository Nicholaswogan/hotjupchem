from setuptools import setup
            
setup(
    name="hotjupchem",
    packages=['hotjupchem'],
    python_requires='>=3.6',
    version="0.1.0", 
    author='Nicholas Wogan',
    author_email = 'nicholaswogan@gmail.com',
    install_requires=[
        'photochem==0.5.6',
        'picaso>=3.2',
        'astropy',
        'matplotlib',
        'numpy',
        'scipy',
        'pyyaml', 
        'numba'
    ],
    include_package_data=True
)
