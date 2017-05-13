package = "torch"
version = "scm-1"

source = {
   url = "git://github.com/torch/torch7.git",
}

description = {
   summary = "Torch7",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/torch7",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "paths >= 1.0",
   "cwrap >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN)
]],
	 platforms = {
      windows = {
           build_command = [[
cmake -E make_directory build && cd build && cmake .. -DINTEL_MKL_DIR=$(INTEL_MKL_DIR) -DINTEL_COMPILER_DIR=$(INTEL_COMPILER_DIR) -DBLAS_LIBRARIES=$(BLAS_LIBRARIES) -DLAPACK_LIBRARIES=$(LAPACK_LIBRARIES) -DLAPACK_FOUND=$(LAPACK_FOUND) -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]]
      }
   },
   install_command = "cd build && $(MAKE) install"
}

