package = 'torch_torch7'
version = 'v1.0.0-1'

source = {
  url = 'git://github.com/cmotl/torch-torch7.git',
  branch = 'v1.0.0',
}

description = {
  summary = 'Torch7',
  detailed = [[
   ]],
  homepage = 'https://github.com/torch/torch7',
  license = 'BSD',
}

dependencies = {
  'lua >= 5.1',
  'torch_cwrap = v1.0.0',
  'torch_paths = v1.0.0',
}

build = {
  type = 'command',
  build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_BINDIR)../lib" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN)
]],
  platforms = {
    windows = {
      build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
    },
  },
  install_command = 'cd build && $(MAKE) install',
}
