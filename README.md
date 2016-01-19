# BING++
[BING++: A Fast High Quality Object Proposal Generator at 100fps](http://arxiv.org/abs/1511.04511)

Tested on Ubuntu 14.04 with CUDA 7.5.

Need to pass OPENCV_PATH which includes lib and include folders to cmake as instructed below.

To build:
```
mkdir build
cd build
cmake .. -DOPENCV_PATH=/path/to/opencv/ (e.g. /home/user/opencv3.0.0/)
make
```

To run:
```
./Objectness/BING++ /path/to/data/ (e.g. /datasets/VOC2007/)
```

Notes:

Annotations.tar.gz has to be extracted into VOC2007 folder.

JPEGImages folder from VOC2007 dataset is not included due to its size.

Included vlfeat for 64bit linux in ext, if you are using another arch please add appropriate libvl or edit CMakelists.txt in Objectness.
