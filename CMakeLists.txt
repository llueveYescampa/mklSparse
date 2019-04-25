cmake_minimum_required(VERSION 3.10)

## set appropriately the following line to create a Double Precision version
set(DOUBLE_PRECISION ON)

project(mklSpmv C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set(MKLROOT /opt/intel/mkl)

## adding the following lines while cleaning the code
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" )
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-result -Wno-unused-parameter -ftree-vectorize -fopt-info-vec-optimized -ffast-math -mavx2")
    # extra flags
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS}  -DMKL_ILP64 -m64 -I${MKLROOT}/include")
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
elseif ( CMAKE_C_COMPILER_ID STREQUAL "Intel" )
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -qopt-report=3 -qopt-report-phase=vec -xCORE-AVX2")
    set(CMAKE_EXE_LINKER_FLAGS "-liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64" )
    #set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -qopt-report=3 -qopt-report-phase=vec ")
elseif ( CMAKE_C_COMPILER_ID STREQUAL "PGI" )
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Minform=inform  -Mnodepchk -Minfo=all -Mvect=simd -DMKL_ILP64 -pgf90libs -I${MKLROOT}/include")
    # extra flags
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS}  -DMKL_ILP64 -pgf90libs -I${MKLROOT}/include")
    set(CMAKE_EXE_LINKER_FLAGS "-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_pgi_thread -lmkl_core -pgf90libs -mp -lpthread -lm -ldl")
endif ( )
## end of adding the following lines while cleaning the code


if (DOUBLE_PRECISION)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DDOUBLE")
endif()


#find_package(MPI REQUIRED)
#include_directories(${MPI_INCLUDE_PATH})

include_directories("src/include" )
add_subdirectory(src)

add_executable(mklSpmv ${mklSparse_Src})
target_link_libraries(mklSpmv  ${CMAKE_EXE_LINKER_FLAGS})


############### Debug messages ####################
#MESSAGE( STATUS "CMAKE_BINARY_DIR: "            ${CMAKE_BINARY_DIR} )
#MESSAGE( STATUS "CMAKE_C_COMPILER_ID: "         ${CMAKE_C_COMPILER_ID} )
#MESSAGE( STATUS "CMAKE_C_FLAGS: "               ${CMAKE_C_FLAGS} )
#MESSAGE( STATUS "CMAKE_CURRENT_SOURCE_DIR: "    ${CMAKE_CURRENT_SOURCE_DIR} )
########### end of Debug messages #################


