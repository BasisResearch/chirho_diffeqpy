import sys
import os.path as osp
from setuptools import find_packages, setup

VERSION = "0.0.1"

MODULE_NAME = "chirho_diffeqpy"

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""


requirements_path = osp.join(osp.dirname(osp.abspath(__file__)), "docs", "source", "requirements.txt")
with open(requirements_path, "r") as f:
    requirements_list = f.read().splitlines()

setup(
    name="chirho_diffeqpy",
    version=VERSION,
    description="Fast backend for chirho dynamical systems",
    long_description=long_description,
    packages=find_packages(include=["chirho_diffeqpy", "chirho_diffeqpy.*"]),
    author="Basis",
    url="https://www.basis.ai/",
    project_urls={
        # "Documentation": "",
        "Source": "https://github.com/BasisResearch/chirho_diffeqpy",
    },
    # Add requirements in docs/source/requirements.txt
    install_requires=requirements_list,
    extras_require={
        "test": [
            "pytest",
            # "pytest-cov",
            # "pytest-xdist",
            "mypy",
            "black",
            "flake8",
            # "isort",
            # "sphinx==7.1.2",
            # "sphinxcontrib-bibtex",
            # "sphinx_rtd_theme==1.3.0",
            # "myst_parser",
            # "nbsphinx",
            # "nbval",
            # "nbqa",
        ],
    },
    python_requires="==3.11.*",  # juliatorch lists the most strict limitation, as it was only tested on 3.11.
    keywords="machine learning statistics probabilistic programming bayesian modeling pytorch",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.11"
    ],
    # yapf
)
