#  LuaFromRocks
#
#  This module finds Lua either by vars set by luarocks if called from luarocks install
#  or from FindLua if not from under rocks.
#
#  Sets variables CMake expects to be set by FindLua:
#
#  LUA_LIBRARIES      - the lua shared library
#  LUA_INCLUDE_DIR    - directory for lua includes
#  LUA_FOUND          - If false, don't attempt to use lua.
#  LUA_VERSION_STRING - Lua version
#  LUA_EXECUTABLE     - Lua executable command

INCLUDE(CheckLibraryExists)

IF(LUA) # if we are using luarocks

  SET(LUA_VERSION_STRING ${LUA_VERSION})

  MESSAGE(STATUS "Lua: using information from luarocks: ${LUA_VERSION}")

  SET(LUA_EXECUTABLE "${LUA}")
  SET(LUA_INCLUDE_DIR "${LUA_INCDIR}")

  IF(LUALIB) # present on windows platforms only
    SET(LUA_LIBRARIES "${LUALIB}")
  ELSE() # too bad, luarocks does not provide it (pfff...)
    GET_FILENAME_COMPONENT(LUA_EXEC_NAME ${LUA_EXECUTABLE} NAME_WE)
    IF(LUA_EXEC_NAME STREQUAL "luajit")
    FIND_LIBRARY(LUA_LIBRARIES
      NAMES luajit luajit-${LUA_VERSION} libluajit  libluajit-${LUA_VERSION}
      PATHS ${LUA_LIBDIR}
      )
    ELSEIF(LUA_EXEC_NAME STREQUAL "lua")
      FIND_LIBRARY(LUA_LIBRARIES
        NAMES lua liblua lua${LUA_VERSION} liblua${LUA_VERSION}
        PATHS ${LUA_LIBDIR}
        )
    ELSE()
      MESSAGE(FATAL_ERROR "You seem to have a non-standard lua installation -- are you using luajit-rocks?")
    ENDIF()
    MESSAGE(STATUS "Lua library guess (no info from luarocks): ${LUA_LIBRARIES}")
  ENDIF()

ELSE() # standalone -- not using luarocks

  IF(WITH_LUA51)
    FIND_PACKAGE(Lua5.1 REQUIRED)
  ELSE()
    # To be extended
    FIND_PACKAGE(Lua REQUIRED)
  ENDIF()

  MESSAGE(STATUS "Lua library: ${LUA_LIBRARIES}")
ENDIF()

IF(LUA_LIBRARIES)
  CHECK_LIBRARY_EXISTS(${LUA_LIBRARIES} luaJIT_setmode "" LUA_JIT)
ENDIF()

MARK_AS_ADVANCED(
  LUA
  LUA_LIBRARIES
  LUA_INCLUDE_DIR
  )

# MESSAGE("LUA_INCLUDE_DIR    is ${LUA_INCDIR}")
# MESSAGE("LUA_VERSION_STRING is ${LUA_VERSION_STRING}")
