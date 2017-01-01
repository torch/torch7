#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.c"
#else

#include "luaG.h"

static int torch_Storage_(new)(lua_State *L)
{
  int index = 1;
  THStorage *storage;
  THAllocator *allocator = luaT_toudata(L, index, "torch.Allocator");
  if (allocator) index++;

  if(lua_type(L, index) == LUA_TSTRING)
  {
    if (allocator)
      THError("Passing allocator not supported when using file mapping");

    const char *fileName = luaL_checkstring(L, index);
    int isShared = 0;
    if(luaT_optboolean(L, index + 1, 0))
      isShared = TH_ALLOCATOR_MAPPED_SHARED;
    ptrdiff_t size = luaL_optinteger(L, index + 2, 0);
    if (isShared && luaT_optboolean(L, index + 3, 0))
      isShared = TH_ALLOCATOR_MAPPED_SHAREDMEM;
    storage = THStorage_(newWithMapping)(fileName, size, isShared);
  }
  else if(lua_type(L, index) == LUA_TTABLE)
  {
    ptrdiff_t size = lua_objlen(L, index);
    ptrdiff_t i;
    if (allocator)
      storage = THStorage_(newWithAllocator)(size, allocator, NULL);
    else
      storage = THStorage_(newWithSize)(size);
    for(i = 1; i <= size; i++)
    {
      lua_rawgeti(L, index, i);
      if(!lua_isnumber(L, -1))
      {
        THStorage_(free)(storage);
        luaL_error(L, "element at index %d is not a number", i);
      }
      THStorage_(set)(storage, i-1, LUA_NUMBER_TO_REAL(lua_tonumber(L, -1)));
      lua_pop(L, 1);
    }
  }
  else if(lua_type(L, index) == LUA_TUSERDATA)
  {
    if (allocator)
      THError("Passing allocator not supported when using storage views");

    THStorage *src = luaT_checkudata(L, index, torch_Storage);
    real *ptr = src->data;
    ptrdiff_t offset = luaL_optinteger(L, index + 1, 1) - 1;
    if (offset < 0 || offset >= src->size) {
      luaL_error(L, "offset out of bounds");
    }
    ptrdiff_t size = luaL_optinteger(L, index + 2, src->size - offset);
    if (size < 1 || size > (src->size - offset)) {
      luaL_error(L, "size out of bounds");
    }
    storage = THStorage_(newWithData)(ptr + offset, size);
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
    storage->view = src;
    THStorage_(retain)(storage->view);
  }
  else if(lua_type(L, index + 1) == LUA_TNUMBER)
  {
    ptrdiff_t size = luaL_optinteger(L, index, 0);
    real *ptr = (real *)luaL_optinteger(L, index + 1, 0);
    if (allocator)
      storage = THStorage_(newWithDataAndAllocator)(ptr, size, allocator, NULL);
    else
      storage = THStorage_(newWithData)(ptr, size);
    storage->flag = TH_STORAGE_REFCOUNTED;
  }
  else
  {
    ptrdiff_t size = luaL_optinteger(L, index, 0);
    if (allocator)
      storage = THStorage_(newWithAllocator)(size, allocator, NULL);
    else
      storage = THStorage_(newWithSize)(size);
  }
  luaT_pushudata(L, storage, torch_Storage);
  return 1;
}

static int torch_Storage_(retain)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THStorage_(retain)(storage);
  return 0;
}

static int torch_Storage_(free)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THStorage_(free)(storage);
  return 0;
}

static int torch_Storage_(resize)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  ptrdiff_t size = luaL_checkinteger(L, 2);
/*  int keepContent = luaT_optboolean(L, 3, 0); */
  THStorage_(resize)(storage, size);/*, keepContent); */
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(copy)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Storage)) )
    THStorage_(copy)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THStorage_(copyByte)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THStorage_(copyChar)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THStorage_(copyShort)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THStorage_(copyInt)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THStorage_(copyLong)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THStorage_(copyFloat)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THStorage_(copyDouble)(storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.HalfStorage")) )
    THStorage_(copyHalf)(storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(fill)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  real value = luaG_(checkreal)(L, 2);
  THStorage_(fill)(storage, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_Storage_(elementSize)(lua_State *L)
{
  luaT_pushinteger(L, THStorage_(elementSize)());
  return 1;
}

static int torch_Storage_(__len__)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  luaT_pushinteger(L, storage->size);
  return 1;
}

static int torch_Storage_(__newindex__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
    ptrdiff_t index = luaL_checkinteger(L, 2) - 1;
    real number = luaG_(checkreal)(L, 3);
    THStorage_(set)(storage, index, number);
    lua_pushboolean(L, 1);
  }
  else
    lua_pushboolean(L, 0);

  return 1;
}

static int torch_Storage_(__index__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
    ptrdiff_t index = luaL_checkinteger(L, 2) - 1;
    luaG_(pushreal)(L, THStorage_(get)(storage, index));
    lua_pushboolean(L, 1);
    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

#if defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_BYTE)
static int torch_Storage_(string)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  if(lua_isstring(L, -1))
  {
    size_t len = 0;
    const char *str = lua_tolstring(L, -1, &len);
    THStorage_(resize)(storage, len);
    memmove(storage->data, str, len);
    lua_settop(L, 1);
  }
  else
    lua_pushlstring(L, (char*)storage->data, storage->size);

  return 1; /* either storage or string */
}
#endif

static int torch_Storage_(totable)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  ptrdiff_t i;

  lua_newtable(L);
  for(i = 0; i < storage->size; i++)
  {
    luaG_(pushreal)(L, storage->data[i]);
    lua_rawseti(L, -2, i+1);
  }
  return 1;
}

static int torch_Storage_(factory)(lua_State *L)
{
  THStorage *storage = THStorage_(new)();
  luaT_pushudata(L, storage, torch_Storage);
  return 1;
}

static int torch_Storage_(write)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

#ifdef DEBUG
  THAssert(storage->size < LONG_MAX);
#endif
  THFile_writeLongScalar(file, storage->size);
  THFile_writeRealRaw(file, storage->data, storage->size);

  return 0;
}

static int torch_Storage_(read)(lua_State *L)
{
  THStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  THFile *file = luaT_checkudata(L, 2, "torch.File");
  ptrdiff_t size = THFile_readLongScalar(file);

  THStorage_(resize)(storage, size);
  THFile_readRealRaw(file, storage->data, storage->size);

  return 0;
}

static const struct luaL_Reg torch_Storage_(_) [] = {
  {"retain", torch_Storage_(retain)},
  {"free", torch_Storage_(free)},
  {"size", torch_Storage_(__len__)},
  {"elementSize", torch_Storage_(elementSize)},
  {"__len__", torch_Storage_(__len__)},
  {"__newindex__", torch_Storage_(__newindex__)},
  {"__index__", torch_Storage_(__index__)},
  {"resize", torch_Storage_(resize)},
  {"fill", torch_Storage_(fill)},
  {"copy", torch_Storage_(copy)},
  {"totable", torch_Storage_(totable)},
  {"write", torch_Storage_(write)},
  {"read", torch_Storage_(read)},
#if defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_BYTE)
  {"string", torch_Storage_(string)},
#endif
  {NULL, NULL}
};

void torch_Storage_(init)(lua_State *L)
{
  luaT_newmetatable(L, torch_Storage, NULL,
                    torch_Storage_(new), torch_Storage_(free), torch_Storage_(factory));
  luaT_setfuncs(L, torch_Storage_(_), 0);
  lua_pop(L, 1);
}

#endif
