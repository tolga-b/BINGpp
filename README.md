# BING++
[BING++: A Fast High Quality Object Proposal Generator at 100fps](http://arxiv.org/abs/1511.04511)

Tested on Ubuntu 14.04 with CUDA 7.5.

Currently has user specific directories in CMakeLists.txt of Objectness.
Need to set OMP_NUM_THREADS to 1 in environmental variables before running (parallel is buggy atm)
A commit is on the way to fix both these issues!

To build:
```
mkdir build
cd build
cmake ..
make
export OMP_NUM_THREADS=1
```
