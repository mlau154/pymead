import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open((HERE / "README.md"), encoding="utf-8") as f:
    README = f.read()

# This call to setup() does all the work
setup(
    name="pymead",
    version="2.0.0",
    description="Generate Bézier-parametrized airfoils and airfoil systems",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mlau154/pymead",
    author="Matthew G Lauer",
    author_email="mlauer2015@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["pymead", "pymead/core", "pymead/symmetric", "pymead/examples", "pymead/utils", "pymead/gui",
              "pymead/tests", "pymead/resources", "pymead/analysis", "pymead/data", "pymead/icons"],
    include_package_data=True,
    install_requires=["scipy", "numpy", "shapely", "matplotlib", "requests", "PyQt5", "pyqtgraph", "python-benedict",
                      "pandas", "dill"],
)
