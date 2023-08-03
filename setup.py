import pathlib
from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('pymead/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open((HERE / "README.md"), encoding="utf-8") as f:
    README = f.read()

# This call to setup() does all the work
setup(
    name="pymead",
    version=main_ns['__version__'],
    description="Python library for generation, aerodynamic analysis, and aerodynamic shape optimization of "
                "Bézier-parametrized airfoils and airfoil systems",
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
    data_files=[('pymead/gui/gui_settings/defaults', ['pymead/gui/gui_settings/defaults/*.json'])],
    packages=["pymead", "pymead/core", "pymead/examples", "pymead/utils", "pymead/gui",
              "pymead/tests", "pymead/resources", "pymead/analysis", "pymead/data", "pymead/icons",
              "pymead/optimization", "pymead/plugins", "pymead/gui/scientificspinbox_master",
              "pymead/gui/pyqt_vertical_tab_widget/pyqt_vertical_tab_widget"],
    include_package_data=True,
    install_requires=["scipy", "numpy", "shapely", "matplotlib", "requests", "PyQt5==5.15.7",
                      "pyqtgraph", "python-benedict", "pandas", "pymoo==0.5.0", "numba"],
)
