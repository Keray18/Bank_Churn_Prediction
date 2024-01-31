#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Keshav Bahoray",
    author_email='keshavbahoray18@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="The bank customer churn dataset is a commonly used dataset for predicting customer churn in the banking industry. This project will predict wether a bank customer will leave the bank or continue to be it's customer. ",
    entry_points={
        'console_scripts': [
            'bank_churn=bank_churn.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bank_churn',
    name='bank_churn',
    packages=find_packages(include=['bank_churn', 'bank_churn.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Keray18/bank_churn',
    version='0.0.1',
    zip_safe=False,
)
