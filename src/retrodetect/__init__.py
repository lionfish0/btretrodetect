"""The docstring for a package (i.e., the docstring of the package’s __init__.py module) should also list the modules and subpackages exported by the package.
"""
from importlib.metadata import version

__version__ = version("retrodetect")

# populate package namespace
from .detect import detectcontact

