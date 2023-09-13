import setuptools

setuptools.setup(
    name="stdata", 
    version="0.0.1",
    author="O Hamelijnck",
    author_email="ohamelijnck@turing.ac.uk",
    description="Standard OPs for space time data and models",
    long_description="",
    long_description_content_type="text/markdown",
    url="N/A",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        "scikit-learn",
        "pandas ",
        "numpy",
        "tqdm"
    ],
    extras_require = {
        'gis': [
            "geopandas",
            "shapely",
        ]
    },
)

