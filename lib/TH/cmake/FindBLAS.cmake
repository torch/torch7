# - Find BLAS library
# This module finds an installed fortran library that implements the BLAS 
# linear-algebra interface (see http://www.netlib.org/blas/).  
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  BLAS_FOUND - set to true if a library implementing the BLAS interface is found.
#  BLAS_INFO - name of the detected BLAS library.
#  BLAS_F2C - set to true if following the f2c return convention
#  BLAS_LIBRARIES - list of libraries to link against to use BLAS
#  BLAS_INCLUDE_DIR - include directory

# Do nothing is BLAS was found before
IF(NOT BLAS_FOUND)

SET(BLAS_LIBRARIES)
SET(BLAS_INCLUDE_DIR)
SET(BLAS_INFO)
SET(BLAS_F2C)

SET(WITH_BLAS "" CACHE STRING "Blas type [mkl/open/goto/acml/atlas/accelerate/veclib/generic]")

# Old FindBlas
INCLUDE(CheckCSourceRuns)
INCLUDE(CheckFortranFunctionExists)

MACRO(Check_Fortran_Libraries LIBRARIES _prefix _name _flags _list)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the 
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to NOTFOUND.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  
  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  message(STATUS "Checking for [${__list}]")

  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      if ( WIN32 )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS ENV LIB 
          PATHS ENV PATH )
      endif ( WIN32 )
      if ( APPLE ) 
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 
          ENV DYLD_LIBRARY_PATH )
      else ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 
          ENV LD_LIBRARY_PATH )
      endif( APPLE )
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      MESSAGE(STATUS "  Library ${_library}: ${${_prefix}_${_library}_LIBRARY}")
    endif(_libraries_work)
  endforeach(_library ${_list})
  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
    if (CMAKE_Fortran_COMPILER_WORKS)
      check_fortran_function_exists(${_name} ${_prefix}${_combined_name}_WORKS)
    else (CMAKE_Fortran_COMPILER_WORKS)
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif (CMAKE_Fortran_COMPILER_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    mark_as_advanced(${_prefix}${_combined_name}_WORKS)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif(_libraries_work)
  if(NOT _libraries_work)
    set(${LIBRARIES} NOTFOUND)
  endif(NOT _libraries_work)
endmacro(Check_Fortran_Libraries)

# Intel MKL?
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "mkl")))
  FIND_PACKAGE(MKL)
  IF(MKL_FOUND)
    SET(BLAS_INFO "mkl")
    SET(BLAS_LIBRARIES ${MKL_LIBRARIES})
    SET(BLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    SET(BLAS_VERSION ${MKL_VERSION})
  ENDIF(MKL_FOUND)
endif()

if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "open")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "openblas")
  if(BLAS_LIBRARIES)
    set(BLAS_INFO "open")
  endif(BLAS_LIBRARIES)
endif()

if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "open")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "openblas;pthread")
  if(BLAS_LIBRARIES)
    set(BLAS_INFO "open")
  endif(BLAS_LIBRARIES)
endif()

if((NOT BLAS_LIBRARIES) AND (WIN32)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "open")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "libopenblas")
  if(BLAS_LIBRARIES)
    set(BLAS_INFO "open")
  endif(BLAS_LIBRARIES)
endif()

if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "goto")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "goto2;gfortran")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "goto")
  endif (BLAS_LIBRARIES)
endif()

if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "goto")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "goto2;gfortran;pthread")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "goto")
  endif (BLAS_LIBRARIES)
endif()

if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "acml")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "acml;gfortran")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "acml")
  endif (BLAS_LIBRARIES)
endif()

# Apple BLAS library?
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "accelerate")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "Accelerate")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "accelerate")
    set(BLAS_IS_ACCELERATE 1)
  endif (BLAS_LIBRARIES)
endif()

if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "veclib")))
  check_fortran_libraries(
    BLAS_LIBRARIES
    BLAS
    sgemm
    ""
    "vecLib")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "veclib")
  endif (BLAS_LIBRARIES)
endif()

# BLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "atlas")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "ptf77blas;atlas;gfortran")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "atlas")
  endif (BLAS_LIBRARIES)
endif()

# Generic BLAS library?
if((NOT BLAS_LIBRARIES)
    AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "generic")))
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "blas")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "generic")
  endif (BLAS_LIBRARIES)
endif()

# Determine if blas was compiled with the f2c conventions
IF (BLAS_LIBRARIES)
  SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  
  CHECK_C_SOURCE_RUNS("
#include <stdlib.h>
#include <stdio.h>
float x[4] = { 1, 2, 3, 4 };
float y[4] = { .1, .01, .001, .0001 };
#ifdef WIN32
  typedef __int64 BLINT;
#else
  typedef long BLINT;
#endif
BLINT four = 4;
BLINT one = 1;
extern double sdot_();
int main() {
  double r = sdot_(&four, x, &one, y, &one);
  exit((float)r != (float).1234);
}" BLAS_F2C_DOUBLE_WORKS )

  CHECK_C_SOURCE_RUNS("
#include <stdlib.h>
#include <stdio.h>
float x[4] = { 1, 2, 3, 4 };
float y[4] = { .1, .01, .001, .0001 };
#ifdef WIN32
  typedef __int64 BLINT;
#else
  typedef long BLINT;
#endif
BLINT four = 4;
BLINT one = 1;
extern float sdot_();
int main() {
  int i;
  double r = sdot_(&four, x, &one, y, &one);
  exit((float)r != (float).1234);
}" BLAS_F2C_FLOAT_WORKS )

  IF (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    MESSAGE(STATUS "This BLAS uses the F2C return conventions")
    SET(BLAS_F2C TRUE)
  ELSE (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    SET(BLAS_F2C FALSE)
  ENDIF (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
ENDIF(BLAS_LIBRARIES)

# epilogue

if(BLAS_LIBRARIES)
  set(BLAS_FOUND TRUE)
else(BLAS_LIBRARIES)
  set(BLAS_FOUND FALSE)
endif(BLAS_LIBRARIES)

IF (NOT BLAS_FOUND AND BLAS_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot find a library with BLAS API. Please specify library location.")
ENDIF (NOT BLAS_FOUND AND BLAS_FIND_REQUIRED)
IF(NOT BLAS_FIND_QUIETLY)
  IF(BLAS_FOUND)
    MESSAGE(STATUS "Found a library with BLAS API (${BLAS_INFO}).")
  ELSE(BLAS_FOUND)
    MESSAGE(STATUS "Cannot find a library with BLAS API. Not using BLAS.")
  ENDIF(BLAS_FOUND)
ENDIF(NOT BLAS_FIND_QUIETLY)

# Do nothing is BLAS was found before
ENDIF(NOT BLAS_FOUND)
