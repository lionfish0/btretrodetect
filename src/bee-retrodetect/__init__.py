from importlib.metadata import version
__version__ = version("bee-retrodetect")

# populate package namespace
from bee-retrodetect.detect import detect, detectcontact
from bee-retrodetect.image_processing import ensemblegetshift, getblockmaxedimage, alignandsubtract
from bee-retrodetect.normxcorr2 import normxcorr2