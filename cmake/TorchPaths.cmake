# workaround another annoying cmake bug
# http://public.kitware.com/Bug/view.php?id=14462
# https://awesome.naquadah.org/bugs/index.php?do=details&task_id=869
MACRO(NORMALIZE_PATH _path_)
  get_filename_component(${_path_}_abs "${${_path_}}" ABSOLUTE)
  SET(${_path_} "${${_path_}_abs}")
ENDMACRO()

#
# This gets standard FindLua variables
#
INCLUDE(LuaFromRocks)

# This is the root of Lua tree that Luarocks used in the build is configured for
# May not be the same as Torch (current) installation directory
GET_FILENAME_COMPONENT(LUA_ROOT "${LUA_BINDIR}" PATH)

NORMALIZE_PATH(CMAKE_INSTALL_PREFIX)
SET(Torch_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

# non-per-moduel install subdirectories: mimic Lua tree, possibly external
# use source Lua variables and LUA_ROOT (which may differ from Torch_INSTALL_PREFIX)
FILE(RELATIVE_PATH Torch_INSTALL_BIN_SUBDIR "${LUA_ROOT}"     "${LUA_BINDIR}")
FILE(RELATIVE_PATH Torch_INSTALL_LIB_SUBDIR "${LUA_ROOT}"     "${LUA_LIBDIR}")
FILE(RELATIVE_PATH Torch_INSTALL_INCLUDE_SUBDIR "${LUA_ROOT}" "${LUA_INCDIR}")

SET(Torch_INSTALL_MAN_SUBDIR "share/man" CACHE PATH
  "Install dir for man pages (relative to Torch_INSTALL_PREFIX)")

SET(Torch_INSTALL_SHARE_SUBDIR "share" CACHE PATH
  "Install dir for data (relative to Torch_INSTALL_PREFIX)")

SET(Torch_INSTALL_CMAKE_SUBDIR "share/cmake/torch" CACHE PATH
  "Install dir for .cmake files (relative to Torch_INSTALL_PREFIX)")

# Per-module install subdirectories: use target Lua variables
FILE(RELATIVE_PATH Torch_INSTALL_LUA_PATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUADIR}")
FILE(RELATIVE_PATH Torch_INSTALL_LUA_CPATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LIBDIR}")
FILE(RELATIVE_PATH Torch_INSTALL_LUA_BINPATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${BINDIR}")
