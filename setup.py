from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

HERE = Path(__file__).parent
README = (HERE / "README.rst").read_text()

setup(
    name='lXtractor',
    version='0.1',
    author='Ivan Reveguk',
    author_email='ivan.reveguk@polytechnique.edu',
    description="Alignment-based patterns' extraction",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
    # install_requires=[
    #     'click~=8.0.1',
    #     'biopython~=1.78',
    #     'ray~=1.4',
    #     'numpy~=1.20',
    #     'pandas~=1.1.4',
    #     'more_itertools~=8.8.0',
    #     'toolz~=0.11.1',
    #     'requests~=2.25.1',
    #     'tqdm~=4.61.0',
    #     'joblib~=1.0.1'
    # ],
    install_requires=[
        'click',
        'biopython',
        'ray',
        'numpy',
        'pandas',
        'more_itertools',
        'toolz',
        'requests',
        'tqdm',
        'joblib'
    ],
    include_package_data=True,
    package_data={
        '': ['*.gz']
    },
    packages=find_packages(exclude=['test']),
    entry_points={
        'console_scripts': [
            'lXtractor = cli:cli',
        ],
    }
)
