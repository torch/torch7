#include <general.h>

#define torch_Generator "torch.Generator"

static const struct luaL_Reg torch_Generator_table_ [] = {
  {NULL, NULL}
};

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

#define torch_Generator_factory torch_Generator_new

void torch_Generator_init(lua_State *L)
{
  luaT_newmetatable(L, torch_Generator, NULL,
                    torch_Generator_new, torch_Generator_free, torch_Generator_factory);
  luaL_register(L, NULL, torch_Generator_table_);
  lua_pop(L, 1);
}
