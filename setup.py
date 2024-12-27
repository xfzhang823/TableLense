""" setup.py file """

from setuptools import setup, find_packages

setup(
    name="TableLense",
    version="0.1",
    packages=find_packages(where="src"),  # Finds all packages in the src directory
    package_dir={
        "": "src"
    },  # Tells setuptools that packages are under the src directory
    include_package_data=True,
    install_requires=[
        "pandas",
        "scikit-learn",
        # Add other dependencies here
    ],
    entry_points={
        "console_scripts": [
            # Define script entry points here if needed
        ],
    },
)
