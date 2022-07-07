from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='crm',
    version='0.1',
    description='A Python implementation of the Capacitance Resistance Models (CRM)',
    long_description=long_description,
    long_description_type="text/x-md",
    url="https://github.com/UT-DIRECT/CRM-scikit-learn",
    packages=setuptools.find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numba", "numpy<=1.22", "scipy", "pandas", "matplotlib", "scikit-learn"]
)
