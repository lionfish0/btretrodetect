    """The docstring for a package (i.e., the docstring of the package’s __init__.py module) should also list the modules and subpackages exported by the package.
    """
from importlib.metadata import version
__version__ = version("bee-retrodetect")

# populate package namespace
from bee-retrodetect.detect import detect, detectcontact
from bee-retrodetect.image_processing import ensemblegetshift, getblockmaxedimage, alignandsubtract
from bee-retrodetect.normxcorr2 import normxcorr2