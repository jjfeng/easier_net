from setuptools import setup, find packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="EASIER-net", # Replace with your own username
    version="0.0.1",
    author="Jean Feng",
    author_email="jeanfeng@gmail.com",
    description="EASIER-net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjfeng/easier_net",
    package_dir={"": "easier_net"},
    packages=find_packages(where=['easier_net']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

