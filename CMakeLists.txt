project( PHOVO )

cmake_minimum_required( VERSION 2.8 )

set(CMAKE_MODULE_PATH ${PHOVO_SOURCE_DIR}/cmake)

# Set the configuration file
configure_file( ${PHOVO_SOURCE_DIR}/config.h.in ${PHOVO_SOURCE_DIR}/phovo/include/config.h )

# Define the output directories for the executables
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PHOVO_BINARY_DIR}/bin )

# Boost library
if(APPLE)
  set( Boost_USE_STATIC_LIBS ON )
else()
  add_definitions( -DBOOST_ALL_DYN_LINK )
endif()
set( Boost_USE_MULTITHREADED ON )
find_package( Boost 1.54 REQUIRED
  COMPONENTS
    filesystem
    system
    program_options
)
include_directories( ${Boost_INCLUDE_DIRS} )

# OpenCV library
find_package( OpenCV 2.4.5 REQUIRED )

# Eigen library
find_package( Eigen3 3.1.3 REQUIRED )
include_directories( ${Eigen_INCLUDE_DIRS} )

# OpenMP
option( PHOVO_WITH_OPENMP "Use OpenMP to enable multithreading" OFF )
if( ${PHOVO_WITH_OPENMP} MATCHES "ON" )
  find_package( OpenMP )
  if( OPENMP_FOUND )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
  else()
    message( "-- Can't find OpenMP. Continuing without it." )
  endif()
endif()

# Ceres solver
option( PHOVO_WITH_CERES "Use Ceres to enable CPhotoconsistencyOdometryCeres" OFF )
if( ${PHOVO_WITH_CERES} MATCHES "ON" )
  find_package( Ceres 1.6.0 REQUIRED )
  include_directories( ${CERES_INCLUDES} )
endif()

# Add the "apps" subdirectory
add_subdirectory( apps )

# Set optimized building:
if( CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native" )
endif()
