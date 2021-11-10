#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', "gensim==3.8.3", "rdkit-pypi==2019.9.2.1"]

test_requirements = ['pytest>=3', ]

setup(
    author="Kin Long Kelvin Lee",
    author_email='kin.long.kelvin.lee@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Doing astrochemistry with robots.",
    entry_points={
        'console_scripts': [
            'astrochem_ml=astrochem_ml.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='astrochem_ml',
    name='astrochem_ml',
    packages=find_packages(include=['astrochem_ml', 'astrochem_ml.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/laserkelvin/astrochem_ml',
    version='0.1.0',
    zip_safe=False,
)
