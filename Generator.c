#include <general.h>

int torch_Generator_new(lua_State *L)
{
  THGenerator *gen = THGenerator_new();
  luaT_pushudata(L, gen, torch_Generator);
  return 1;
}

int torch_Generator_free(lua_State *L)
{
  THGenerator *gen= luaT_checkudata(L, 1, torch_Generator);
  THGenerator_free(gen);
  return 0;
}

static int torch_Generator_write(lua_State *L)
{
  THGenerator *gen = luaT_checkudata(L, 1, torch_Generator);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  THFile_writeByteRaw(file, (unsigned char *)gen, sizeof(THGenerator));
  return 0;
}

static int torch_Generator_read(lua_State *L)
{
  THGenerator *gen = luaT_checkudata(L, 1, torch_Generator);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  THFile_readByteRaw(file, (unsigned char *)gen, sizeof(THGenerator));
  return 0;
}


static const struct luaL_Reg torch_Generator_table_ [] = {
  {"write", torch_Generator_write},
  {"read", torch_Generator_read},
  {NULL, NULL}
};

#define torch_Generator_factory torch_Generator_new

void torch_Generator_init(lua_State *L)
{
  luaT_newmetatable(L, torch_Generator, NULL,
                    torch_Generator_new, torch_Generator_free, torch_Generator_factory);
  luaT_setfuncs(L, torch_Generator_table_, 0);
  lua_pop(L, 1);
}
