from distutils.core import setup
setup(
  name = 'retrodetect',
  packages = ['retrodetect'],
  version = '2.0',
  description = 'Finds retroreflectors in a series of flash illuminated photos',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/retrodetect.git',
  download_url = 'https://github.com/lionfish0/retrodetect.git',
  keywords = ['image processing','retroreflectors'],
  classifiers = [],
  install_requires=['numpy'],
  scripts=['bin/btretrodetect'],
)
