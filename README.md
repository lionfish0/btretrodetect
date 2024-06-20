# retrodetect
- This new version (completely rewritten) assumes all the photos are flash photos.

# install
pip install git+https://github.com/lionfish0/retrodetect.git

# usage
One passes a path and the tool will recursively search through the subdirectories, finding all the images, sorting them (within that folder) and applying the retrodetect algorithm.

Example:
```btretrodetect ~/Documents/Research/rsync_bee/test/beephotos/2023-06-29/sessionA/setA/cam5/02D49670796/ --after 10:32:29 --before 10:33:29 --threshold -10```

```usage: btretrodetect [-h] [--after AFTER] [--before BEFORE] [--refreshcache] [--threshold THRESHOLD] [--sourcename SOURCENAME] imgpath

Runs the retoreflector detection algorithm

positional arguments:
  imgpath               Path to images (it will recursively search for images in these paths)

options:
  -h, --help            show this help message and exit
  --after AFTER         Only process images that were created after this time HH:MM:SS
  --before BEFORE       Only process images that were created before this time HH:MM:SS
  --refreshcache        Whether to refresh the cache
  --threshold THRESHOLD
                        Threshold of score before adding to data
  --sourcename SOURCENAME
                        The name to give this source of labels (default:retrodetect)```
