"""
SVD Steg python modules.

Colin Page <cwpage@umich.edu>
"""

from setuptools import setup

setup(
    name='svd_steg',
    version='0.1.0',
    packages=['svd_steg'],
    include_package_data=True,
    install_requires=[
        'click==6.7',
        'pylint==2.1.1',
        'pydocstyle==2.0.0',
        'pycodestyle==2.3.1',
        'numpy==1.15.3',
        'imageio==2.4.1',
    ],
    entry_points={
        'console_scripts': [
            'svd_steg = svd_steg.__main__:main',
        ]
    },
)
