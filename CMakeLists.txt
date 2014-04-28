cmake_minimum_required(VERSION 2.8)
project( DoF_Renderer )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
include_directories ( ${CUDA_TOOLKIT_INCLUDE} )
include_directories ( "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc" )

# set source files
set(PROJECT_SRC src/test_dof_renderer.cpp src/dof_renderer.cpp src/debugging_functions.cpp)
set(PROJECT_INC include/convolutionSeparable_common.h include/dof_renderer.hpp include/debugging_functions.hpp)
set(TARGET_EXECUTABLE test_dof_renderer)
# set(PROJECT_SRC convolutionSeparable_common.h)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

cuda_add_executable( ${TARGET_EXECUTABLE} ${PROJECT_SRC} ${PROJECT_INC} OPTIONS ${GENCODE})

target_link_libraries( ${TARGET_EXECUTABLE} ${OpenCV_LIBS} )
target_link_libraries( ${TARGET_EXECUTABLE} ${CUDA_LIBS} )
target_link_libraries( ${TARGET_EXECUTABLE} ${CUFFT_LIBRARY})

# cuda_add_executable(convolutionSeparable ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})

# install(TARGETS convolutionSeparable
        # RUNTIME DESTINATION ${INSTALL_PATH})
