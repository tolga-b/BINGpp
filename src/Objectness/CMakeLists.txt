#
find_package(CUDA REQUIRED)
include(FindCUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_30,code=sm_30)

set(INC_DIRS ${BING++_SOURCE_DIR}/LibLinear ${BING++_SOURCE_DIR}/LibLinear/blas ${BING++_SOURCE_DIR}/../ext /usr/local/cuda/include ${OPENCV_PATH}include ${BING++_SOURCE_DIR}/Objectness/gSLIC_Lib ${BING++_SOURCE_DIR}/Objectness/FelzenSegment )
set(LIB_DIRS ${BING++_BINARY_DIR}/LibLinear ${BING++_SOURCE_DIR}/../ext ${OPENCV_PATH}lib)

add_subdirectory(ORUtils)

set(GSLIC_LIB
gSLIC_Lib/engines/gSLIC_core_engine.h
gSLIC_Lib/engines/gSLIC_seg_engine.h
gSLIC_Lib/engines/gSLIC_seg_engine_GPU.h
gSLIC_Lib/engines/gSLIC_seg_engine_shared.h
gSLIC_Lib/engines/gSLIC_core_engine.cpp
gSLIC_Lib/engines/gSLIC_seg_engine.cpp
gSLIC_Lib/engines/gSLIC_seg_engine_GPU.cu
gSLIC_Lib/objects/gSLIC_settings.h
gSLIC_Lib/objects/gSLIC_spixel_info.h
gSLIC_Lib/gSLIC_defines.h
gSLIC_Lib/gSLIC.h
)

SOURCE_GROUP(engines FILES ${GSLIC_LIB})

#OPENCV_DIR should be set as an environment variable in the shell if it is not
#present in the standard directories

# message(STATUS ${LIB_DIRS})
include_directories(${INC_DIRS})
link_directories(${LIB_DIRS})

if (${CMAKE_COMPILER_IS_GNUCXX})
  set(OPENMP_FLAG "-fopenmp -std=c++11 -ftree-vectorize")
endif (${CMAKE_COMPILER_IS_GNUCXX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG} -O3") # -O0 -pg

set(SRC_FILES CmFile.cpp CmShow.cpp DataSetVOC.cpp FilterBING.cpp mtse.cpp Objectness.cpp stdafx.cpp)
cuda_add_library(gSLIC_lib
			${GSLIC_LIB})
add_library(objectness ${SRC_FILES})
cuda_add_executable(${PROJECT_NAME} ${SRC_FILES} kernel.cu)
target_link_libraries(${PROJECT_NAME} opencv_highgui opencv_core opencv_imgproc opencv_ts opencv_imgcodecs liblinear blas gSLIC_lib ${CUDA_LIBRARY} vl)
