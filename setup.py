from distutils.core import setup
setup(
  name = 'retrodetect',
  packages = ['retrodetect'],
  version = '1.01',
  description = 'Finds retroreflectors in a pair of images',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/retrodetect.git',
  download_url = 'https://github.com/lionfish0/retrodetect.git',
  keywords = ['image processing','retroreflectors'],
  classifiers = [],
  install_requires=['numpy'],
)
