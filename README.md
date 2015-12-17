# BINGpp
BING++: A Fast High Quality Object Proposal Generator at 100fps

Currently has user specific directories in CMakeLists.txt of Objectness folder.
Tested on Ubuntu 14.04 with CUDA 7.5
Need to set OMP_NUM_THREADS to 1 in environmental variables before running (parallel is buggy atm)
mkdir build
cd build
cmake ..
make
