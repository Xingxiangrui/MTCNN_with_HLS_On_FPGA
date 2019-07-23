cmake_minimum_required(VERSION 2.8)
project(main)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(QMAKE_CXXFLAGS "-std=c++11")
#include_directories(/1t_second/xxr2/c_mtcnn_head/include/)
#link_directories(/1t_second/xxr2/c_mtcnn_head/lib/)

find_package( OpenCV REQUIRED)

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${DIR_SRCS})
target_link_libraries( main ${OpenCV_LIBS} )
#target_link_libraries( main openblas)
