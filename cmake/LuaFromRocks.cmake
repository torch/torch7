#  LuaFromRocks
#
#  This module finds Lua either by:
#  - vars set by luarocks if called from luarocks install
#  - from target isntall dirs if builing our own Lua
#  - FindLua() otherwise
#
#  Sets variables CMake expects to be set by FindLua:
#
#  LUA_LIBRARIES      - the lua shared library
#  LUA_INCLUDE_DIR    - directory for lua includes
#  LUA_FOUND          - If false, don't attempt to use lua.
#  LUA_VERSION_STRING - Lua version
#  LUA_EXECUTABLE     - Lua executable command
#
#  Makes sure LuaRocks variables set even if we are not using LuaRocks:
#  LUA_INCDIR - place where lua headers exist
#  LUA_LIBDIR - place where lua libraries exist
#  LUA_BINDIR - place where lua libraries exist
#  SCRIPTS_DIR - usually, same as LUA_BINDIR
#  LUADIR - LUA_PATH
#  LIBDIR - LUA_CPATH
#  LUALIB - the lua library to link against

INCLUDE(CheckLibraryExists)

SET(LUA_EXECUTABLE "${LUA}")

IF(LUA) # if we are using luarocks

  SET(LUA_VERSION_STRING ${LUA_VERSION})

  MESSAGE(STATUS "Lua: using information from luarocks: ${LUA_VERSION}")

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
  IF(WITH_NATIVE_LUA)
    IF(WITH_LUA51)
      FIND_PACKAGE(Lua5.1 REQUIRED)
    ELSE()
      # To be extended
      FIND_PACKAGE(Lua REQUIRED)
    ENDIF()
  ELSE()
    IF (NOT DEFINED ${LUA_INCDIR})
      SET(LUA_INCDIR ${CMAKE_INSTALL_PREFIX}/include)
      MESSAGE(STATUS "LUA_INCDIR: ${LUA_INCDIR}")
    ENDIF(NOT DEFINED ${LUA_INCDIR})
    IF (NOT DEFINED ${LUA_LIBDIR})
      SET(LUA_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
      MESSAGE(STATUS "LUA_LIBDIR: ${LUA_LIBDIR}")
    ENDIF(NOT DEFINED ${LUA_LIBDIR})
    IF (NOT DEFINED ${LUA_BINDIR})
      SET(LUA_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
      MESSAGE(STATUS "LUA_BINDIR: ${LUA_BINDIR}")
    ENDIF(NOT DEFINED ${LUA_BINDIR})
    IF (NOT DEFINED ${LUALIB})
      IF (WITH_LUAJIT21 OR WITH_LUAJIT20)
        SET(LUALIB ${LIBRARY_OUTPUT_PATH}/libluajit.so)
      ELSE (WITH_LUAJIT21 OR WITH_LUAJIT20)
        SET(LUALIB ${LIBRARY_OUTPUT_PATH}/liblua.so)
      ENDIF (WITH_LUAJIT21 OR WITH_LUAJIT20)
      MESSAGE(STATUS "LUALIB: ${LUALIB}")
    ENDIF(NOT DEFINED ${LUALIB})
    IF (NOT DEFINED ${LUADIR})
      SET(LUADIR ${CMAKE_INSTALL_PREFIX}/${INSTALL_LUA_PATH_SUBDIR})
      MESSAGE(STATUS "LUADIR: ${LUADIR}")
    ENDIF(NOT DEFINED ${LUADIR})
    IF (NOT DEFINED ${LIBDIR})
      SET(LIBDIR ${CMAKE_INSTALL_PREFIX}/${INSTALL_LUA_CPATH_SUBDIR})
      MESSAGE(STATUS "LIBDIR: ${LIBDIR}")
    ENDIF(NOT DEFINED ${LIBDIR})
    IF (NOT DEFINED ${LUA_BINDIR})
      SET(LUA_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
    ENDIF(NOT DEFINED ${LUA_BINDIR})
    IF (NOT DEFINED ${SCRIPTS_DIR})
      SET(SCRIPTS_DIR ${CMAKE_INSTALL_PREFIX}/bin)
    ENDIF(NOT DEFINED ${SCRIPTS_DIR})
    IF (WITH_LUAJIT21 OR WITH_LUAJIT20)
      SET(LUA "luajit")
    ELSE (WITH_LUAJIT21 OR WITH_LUAJIT20)
      SET(LUA "lua")
    ENDIF (WITH_LUAJIT21 OR WITH_LUAJIT20)
  ENDIF(WITH_NATIVE_LUA)
ENDIF()

NORMALIZE_PATH(LUA_BINDIR)
NORMALIZE_PATH(LUA_LIBDIR)
NORMALIZE_PATH(LUA_INCDIR)

NORMALIZE_PATH(LUADIR)
NORMALIZE_PATH(LIBDIR)
NORMALIZE_PATH(BINDIR)

SET(LUA_EXECUTABLE "${LUA}")
SET(LUA_INCLUDE_DIR "${LUA_INCDIR}")

MESSAGE(STATUS "LUA_BINDIR: ${LUA_BINDIR}")
MESSAGE(STATUS "SCRIPTS_DIR: ${SCRIPTS_DIR}")
MESSAGE(STATUS "Lua library: ${LUA_LIBRARIES}")

INCLUDE_DIRECTORIES(${LUA_INCDIR})

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
