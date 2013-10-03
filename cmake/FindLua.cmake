# Find the Lua includes and library
#
# LUA_INCLUDE_DIR -- where to find the includes
# LUA_LIBRARIES -- list of libraries to link against
# LUA_EXECUTABLE -- lua executable
# LUA_FOUND -- set to 1 if found

SET(LUA_INCLUDE_DIR "${LUA_INCDIR}")
SET(LUA_EXECUTABLE "${LUA_BINDIR}/${LUA}")

FIND_LIBRARY(LUA_LIBRARIES
             ${LUA}
             HINTS ${LUA_LIBDIR}
             NO_CMAKE_PATH
             NO_CMAKE_ENVIRONMENT PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

IF(NOT LUA_LIBRARIES)
  SET(LUA_FOUND 0)
ELSE()
  SET(LUA_FOUND 1)
ENDIF()
