cmake_minimum_required(VERSION 3.13)

## set appropriately the following line to create a Double Precision version
set(DOUBLE_PRECISION ON)

project(mklSpmv C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(MKL CONFIG REQUIRED)
#message(STATUS "${MKL_IMPORTED_TARGETS}")
#message(STATUS "                                    ${MKL_ROOT}")
#message(STATUS "                                    ${MKL_ARCH}")
#message(STATUS "                                    ${MKL_THREADING}")

## adding the following lines while cleaning the code
MESSAGE( STATUS "CMAKE_C_COMPILER_ID: "         ${CMAKE_C_COMPILER_ID} )
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" )
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-result -Wno-unused-parameter -ftree-vectorize -fopt-info-vec-optimized -ffast-math -mavx2")
    set(MKL_LIBRARIE_MISSING "-L${MKL_ROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lm")
    #message(STATUS                              "${MKL_LIBRARIE_MISSING}")
elseif ( CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" )
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -fvectorize  -qopt-report-file=stderr -qopt-report=max  -xCORE-AVX2")
    #set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -qopt-report=5 -qopt-report-phase=vec -xCORE-AVX2")
    set(MKL_LIBRARIE_MISSING "-L${MKL_ROOT}/lib/intel64 -lmkl_intel_lp64")
    #message(STATUS                              "${MKL_LIBRARIE_MISSING}")
elseif ( CMAKE_C_COMPILER_ID STREQUAL "NVHPC" )
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Minform=inform  -Mnodepchk -Minfo=all -Mvect=simd")
    set(MKL_LIBRARIE_MISSING "-L${MKL_ROOT}/lib/intel64 -lmkl_intel_lp64")
    #message(STATUS                              "${MKL_LIBRARIE_MISSING}")
endif ( )
## end of adding the following lines while cleaning the code


#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


if (DOUBLE_PRECISION)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DDOUBLE")
endif()

#include_directories("src/include" )
include_directories(${MKL_INCLUDE}; "src/include")
add_subdirectory(src)

add_executable(mklSpmv ${mklSparse_Src})
#target_link_libraries(mklSpmv  MKL::MKL)
target_link_libraries(mklSpmv  ${MKL_LIBRARIE_MISSING} MKL::mkl_core;MKL::mkl_intel_thread)

############### Debug messages ####################
#MESSAGE( STATUS "CMAKE_BINARY_DIR: "            ${CMAKE_BINARY_DIR} )
#MESSAGE( STATUS "CMAKE_C_COMPILER_ID: "         ${CMAKE_C_COMPILER_ID} )
#MESSAGE( STATUS "CMAKE_C_FLAGS: "               ${CMAKE_C_FLAGS} )
#MESSAGE( STATUS "CMAKE_CURRENT_SOURCE_DIR: "    ${CMAKE_CURRENT_SOURCE_DIR} )
########### end of Debug messages #################
