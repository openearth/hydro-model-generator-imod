#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Huite Bootsma",
    author_email='huite.bootsma@deltares.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Generator of model files for Hydro Model Builder",
    install_requires=requirements,
    license="MIT license",
    long_description=readme, # + '\n\n' + history,
    include_package_data=True,
    keywords='hydro_model_generator_imod',
    name='hydro_model_generator_imod',
    packages=find_packages(include=['hydro_model_generator_imod']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/openearth/hydro-model-generator-imod',
    version='0.1.0',
    zip_safe=False,
)
