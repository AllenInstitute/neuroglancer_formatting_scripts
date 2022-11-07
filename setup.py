from setuptools import setup, find_packages


setup(
    name="neuroglancer_formatting_scripts",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
