from setuptools import setup, find_packages

setup(
    name='trees_classifiers', 
    
    version='0.1.0',
    
    author='Arthur',
    author_email='arthurcarvalhorodrigues2409@gmail.com',
    
    description='lista 05',
    
    packages=find_packages(),
    
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn' 
    ],
    
    keywords=['decision tree', 'id3', 'c45', 'cart',],
    
    python_requires='>=3.8',
)
