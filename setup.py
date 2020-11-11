import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuro-helper",
    version="0.0.8",
    author="Mehrshad Golesorkhi",
    author_email="mehrshad.golesorkhi@gmail.com",
    description="A simple package dealing with some analysis in my PhD thesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mehrshadg/neuro-data-helper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
