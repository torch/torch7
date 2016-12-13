#include "general.h"
#include "utils.h"

#ifdef WIN32
# include <time.h>
#else
# include <sys/time.h>
#endif

THLongStorage* torch_checklongargs(lua_State *L, int index)
{
  THLongStorage *storage;
  int i;
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, "torch.LongStorage"))
  {
    THLongStorage *storagesrc = luaT_toudata(L, index, "torch.LongStorage");
    storage = THLongStorage_newWithSize(storagesrc->size);
    THLongStorage_copy(storage, storagesrc);
  }
  else
  {
    storage = THLongStorage_newWithSize(narg);
    for(i = index; i < index+narg; i++)
    {
      if(!lua_isnumber(L, i))
      {
        THLongStorage_free(storage);
        luaL_argerror(L, i, "number expected");
      }
      THLongStorage_set(storage, i-index, lua_tonumber(L, i));
    }
  }
  return storage;
}

int torch_islongargs(lua_State *L, int index)
{
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, "torch.LongStorage"))
  {
    return 1;
  }
  else
  {
    int i;

    for(i = index; i < index+narg; i++)
    {
      if(!lua_isnumber(L, i))
        return 0;
    }
    return 1;
  }
  return 0;
}

#ifdef _WIN32
#include <windows.h>
#include <io.h>
static __declspec( thread ) LARGE_INTEGER ticksPerSecond = { 0 };
#endif

static int torch_isatty(lua_State *L)
{
  FILE **fp = (FILE **) luaL_checkudata(L, -1, LUA_FILEHANDLE);
#ifdef _WIN32
  lua_pushboolean(L, _isatty(_fileno(*fp)));
#else
  lua_pushboolean(L, isatty(fileno(*fp)));
#endif
  return 1;
}

static double real_time()
{
#ifdef _WIN32
  if (ticksPerSecond.QuadPart == 0)
  {
    QueryPerformanceFrequency(&ticksPerSecond);
  }
  LARGE_INTEGER current;
  QueryPerformanceCounter(&current);
  return (double)(current.QuadPart) / ticksPerSecond.QuadPart;
#else
  struct timeval current;
  gettimeofday(&current, NULL);
  return (current.tv_sec + current.tv_usec/1000000.0);
#endif
}

static int torch_lua_tic(lua_State* L)
{
  double ttime = real_time();
  lua_pushnumber(L,ttime);
  return 1;
}

static int torch_lua_toc(lua_State* L)
{
  double toctime = real_time();
  lua_Number tictime = luaL_checknumber(L,1);
  lua_pushnumber(L,toctime-tictime);
  return 1;
}

static int torch_lua_getdefaulttensortype(lua_State *L)
{
  const char* tname = torch_getdefaulttensortype(L);
  if(tname)
  {
    lua_pushstring(L, tname);
    return 1;
  }
  return 0;
}

const char* torch_getdefaulttensortype(lua_State *L)
{
  lua_getglobal(L, "torch");
  if(lua_istable(L, -1))
  {
    lua_getfield(L, -1, "Tensor");
    if(lua_istable(L, -1))
    {
      if(lua_getmetatable(L, -1))
      {
        lua_pushstring(L, "__index");
        lua_rawget(L, -2);
        if(lua_istable(L, -1))
        {
          lua_rawget(L, LUA_REGISTRYINDEX);
          if(lua_isstring(L, -1))
          {
            const char *tname = lua_tostring(L, -1);
            lua_pop(L, 4);
            return tname;
          }
        }
        else
        {
          lua_pop(L, 4);
          return NULL;
        }
      }
      else
      {
        lua_pop(L, 2);
        return NULL;
      }
    }
    else
    {
      lua_pop(L, 2);
      return NULL;
    }
  }
  else
  {
    lua_pop(L, 1);
    return NULL;
  }
  return NULL;
}

static int torch_getnumthreads(lua_State *L)
{
  lua_pushinteger(L, THGetNumThreads());
  return 1;
}

static int torch_setnumthreads(lua_State *L)
{
  THSetNumThreads(luaL_checkint(L, 1));
  return 0;
}

static int torch_getnumcores(lua_State *L)
{
  lua_pushinteger(L, THGetNumCores());
  return 1;
}

static void luaTorchGCFunction(void *data)
{
  lua_State *L = data;
  lua_gc(L, LUA_GCCOLLECT, 0);
}

static int torch_setheaptracking(lua_State *L)
{
  int enabled = luaT_checkboolean(L,1);
  lua_getglobal(L, "torch");
  lua_pushboolean(L, enabled);
  lua_setfield(L, -2, "_heaptracking");
  if(enabled) {
    THSetGCHandler(luaTorchGCFunction, L);
  } else {
    THSetGCHandler(NULL, NULL);
  }
  return 0;
}

static void luaTorchErrorHandlerFunction(const char *msg, void *data)
{
  lua_State *L = data;
  luaL_error(L, msg);
}

static void luaTorchArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
{
  lua_State *L = data;
  luaL_argcheck(L, 0, argNumber, msg);
}

static int torch_updateerrorhandlers(lua_State *L)
{
  THSetErrorHandler(luaTorchErrorHandlerFunction, L);
  THSetArgErrorHandler(luaTorchArgErrorHandlerFunction, L);
  return 0;
}

static const struct luaL_Reg torch_utils__ [] = {
  {"getdefaulttensortype", torch_lua_getdefaulttensortype},
  {"isatty", torch_isatty},
  {"tic", torch_lua_tic},
  {"toc", torch_lua_toc},
  {"setnumthreads", torch_setnumthreads},
  {"getnumthreads", torch_getnumthreads},
  {"getnumcores", torch_getnumcores},
  {"factory", luaT_lua_factory},
  {"getconstructortable", luaT_lua_getconstructortable},
  {"typename", luaT_lua_typename},
  {"isequal", luaT_lua_isequal},
  {"getenv", luaT_lua_getenv},
  {"setenv", luaT_lua_setenv},
  {"newmetatable", luaT_lua_newmetatable},
  {"setmetatable", luaT_lua_setmetatable},
  {"getmetatable", luaT_lua_getmetatable},
  {"metatype", luaT_lua_metatype},
  {"pushudata", luaT_lua_pushudata},
  {"version", luaT_lua_version},
  {"pointer", luaT_lua_pointer},
  {"setheaptracking", torch_setheaptracking},
  {"updateerrorhandlers", torch_updateerrorhandlers},
  {NULL, NULL}
};

void torch_utils_init(lua_State *L)
{
  torch_updateerrorhandlers(L);
  luaT_setfuncs(L, torch_utils__, 0);
}
