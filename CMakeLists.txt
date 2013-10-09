CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  "${CMAKE_MODULE_PATH}")

INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)
INCLUDE(TorchExports)

# Torch libraries
ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${LUA_INCDIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/TH")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/lib/TH")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/luaT")
LINK_DIRECTORIES("${LUA_LIBDIR}")

# Flags
# When using MSVC
IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
ENDIF(MSVC)

# ARM specific flags
FIND_PACKAGE(ARM)
IF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__NEON__ ${CMAKE_C_FLAGS}")
ENDIF (NEON_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)

SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c Timer.c utils.c init.c TensorOperator.c TensorMath.c random.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua Tester.lua test/test.lua)

# Necessary do generate wrapper
ADD_TORCH_WRAP(tensormathwrap TensorMath.lua)
ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(torch luaT TH)

IF(LUALIB)
  TARGET_LINK_LIBRARIES(torch ${LUALIB})
ENDIF()
