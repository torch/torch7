#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "luaT.h"

void* luaT_alloc(lua_State *L, ptrdiff_t size)
{
  void *ptr;

  if(size == 0)
    return NULL;

  if(size < 0)
    luaL_error(L, "$ Torch: invalid memory size -- maybe an overflow?");

  ptr = malloc(size);
  if(!ptr)
    luaL_error(L, "$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", size/1073741824);

  return ptr;
}

void* luaT_realloc(lua_State *L, void *ptr, ptrdiff_t size)
{
  if(!ptr)
    return(luaT_alloc(L, size));

  if(size == 0)
  {
    luaT_free(L, ptr);
    return NULL;
  }

  if(size < 0)
    luaL_error(L, "$ Torch: invalid memory size -- maybe an overflow?");

  ptr = realloc(ptr, size);
  if(!ptr)
    luaL_error(L, "$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);
  return ptr;
}

void luaT_free(lua_State *L, void *ptr)
{
  free(ptr);
}

void luaT_setfuncs(lua_State *L, const luaL_Reg *l, int nup)
{
#if LUA_VERSION_NUM == 501
  luaL_checkstack(L, nup+1, "too many upvalues");
  for (; l->name != NULL; l++) {  /* fill the table with given functions */
    int i;
    lua_pushstring(L, l->name);
    for (i = 0; i < nup; i++)  /* copy upvalues to the top */
      lua_pushvalue(L, -(nup+1));
    lua_pushcclosure(L, l->func, nup);  /* closure with those upvalues */
    lua_settable(L, -(nup + 3));
  }
  lua_pop(L, nup);  /* remove upvalues */
#else
  luaL_setfuncs(L, l, nup);
#endif
}

void luaT_stackdump(lua_State *L)
{
  int i;
  const char *tname = NULL;
  int top = lua_gettop(L);
  for(i = 1; i <= top; i++)
  {
    int t = lua_type(L, i);
    printf("%3d. ", i);
    switch(t)
    {
      case LUA_TSTRING:
        printf("'%s'", lua_tostring(L,i));
        break;
      case LUA_TBOOLEAN:
        printf(lua_toboolean(L, i) ? "true" : "false");
        break;
      case LUA_TNUMBER:
        printf("%g", lua_tonumber(L,i));
        break;
      case LUA_TUSERDATA:
        tname = luaT_typename(L, i);
        printf("userdata %p [%s]", lua_topointer(L, i), (tname ? tname : "not a Torch object"));
        break;
      case 10:
        tname = luaT_typename(L, i);
        printf("cdata %p [%s]", lua_topointer(L, i), (tname ? tname : "not a Torch object"));
        break;
      case LUA_TTABLE:
        lua_pushvalue(L, i);
        lua_rawget(L, LUA_REGISTRYINDEX);
        if(lua_isstring(L, -1))
          tname = lua_tostring(L, -1); /*luaT_typenameid(L, lua_tostring(L, -1)); */
        else
          tname = NULL;
        lua_pop(L, 1);
        if(tname)
          printf("metatable [%s]", tname);
        else
        {
          tname = luaT_typename(L, i);
          printf("table %p [%s]", lua_topointer(L, i), (tname ? tname : "not a Torch object"));
        }
        break;
      default:
        printf("Lua object type: %s", lua_typename(L,t));
        break;
    }
    printf("\n");
  }
  printf("---------------------------------------------\n");
}

/* metatable operator methods */
static int luaT_mt__index(lua_State *L);
static int luaT_mt__newindex(lua_State *L);
static int luaT_mt__tostring(lua_State *L);
static int luaT_mt__add(lua_State *L);
static int luaT_mt__sub(lua_State *L);
static int luaT_mt__mul(lua_State *L);
static int luaT_mt__div(lua_State *L);
static int luaT_mt__mod(lua_State *L);
static int luaT_mt__pow(lua_State *L);
static int luaT_mt__unm(lua_State *L);
static int luaT_mt__concat(lua_State *L);
static int luaT_mt__len(lua_State *L);
static int luaT_mt__eq(lua_State *L);
static int luaT_mt__lt(lua_State *L);
static int luaT_mt__le(lua_State *L);
static int luaT_mt__call(lua_State *L);

/* Constructor-metatable methods */
static int luaT_cmt__call(lua_State *L);
static int luaT_cmt__newindex(lua_State *L);

const char* luaT_newmetatable(lua_State *L, const char *tname, const char *parent_tname,
                              lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory)
{
  return luaT_newlocalmetatable(L, tname, parent_tname,
                                constructor, destructor, factory, 0);
}

const char* luaT_newlocalmetatable(lua_State *L, const char *tname, const char *parent_tname,
                                   lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory, int moduleidx)
{
  lua_pushcfunction(L, luaT_lua_newmetatable);
  lua_pushstring(L, tname);
  (parent_tname ? (void)lua_pushstring(L, parent_tname) : lua_pushnil(L));
  (constructor ? lua_pushcfunction(L, constructor) : lua_pushnil(L));
  (destructor ? lua_pushcfunction(L, destructor) : lua_pushnil(L));
  (factory ? lua_pushcfunction(L, factory) : lua_pushnil(L));
  (moduleidx > 0 ? lua_pushvalue(L, moduleidx) : lua_pushnil(L));
  lua_call(L, 6, 1);
  return luaT_typenameid(L, tname);
}

int luaT_pushmetatable(lua_State *L, const char *tname)
{
  lua_getfield(L, LUA_REGISTRYINDEX, tname);
  if(lua_isnil(L, -1))
  {
    lua_pop(L, 1);
    return 0;
  }
  return 1;
}

const char *luaT_typenameid(lua_State *L, const char *tname)
{
  if(luaT_pushmetatable(L, tname))
  {
    const char *tnameid = NULL;
    lua_rawget(L, LUA_REGISTRYINDEX);
    if(lua_isstring(L, -1))
      tnameid = lua_tostring(L, -1);
    lua_pop(L, 1); /* the string/nil */
    return tnameid;
  }
  return NULL;
}

static const char cdataname[] = ""
  "local ok, ffi = pcall(require, 'ffi')\n"
  "if ok then\n"
  "  local id2name = {}\n"
  "  return function(cdata, name)\n"
  "    local id\n"
  "    if jit then\n"
  "      id = tonumber(ffi.typeof(cdata))\n"
  "    else\n"
  "      id = tostring(ffi.typeof(cdata))\n"
  "    end\n"
  "    if id then\n"
  "      if name then\n"
  "        id2name[id] = name\n"
  "        return name\n"
  "      else\n"
  "        return rawget(id2name, id)\n"
  "      end\n"
  "    end\n"
  "    return nil\n"
  "  end\n"
  "else\n"
  "  return function() end\n"
  "end\n";

static const char* luaT_cdataname(lua_State *L, int ud, const char *tname)
{
  lua_pushstring(L, "__cdataname");
  lua_rawget(L, LUA_REGISTRYINDEX);
  if(lua_isnil(L,-1))
  {
    lua_pop(L, 1);

    if(luaL_dostring(L, cdataname)) /* did something go wrong? */
      luaL_error(L, "internal error (could not load cdataname): %s", lua_tostring(L, -1));

    lua_pushstring(L, "__cdataname");
    lua_pushvalue(L, -2);
    lua_rawset(L, LUA_REGISTRYINDEX);
  }
  if(!lua_isfunction(L, -1)) /* should not happen */
    luaL_error(L, "internal error (cdataname is not a function)");

  lua_pushvalue(L, ud);
  if(tname)
    lua_pushstring(L, tname);
  if(lua_pcall(L, (tname ? 2 : 1), 1, 0))
    luaL_error(L, "internal error (cdataname): %s", lua_tostring(L, -1));

  tname = lua_tostring(L, -1);
  lua_pop(L, 1);

  return tname;
}

static void* CDATA_MT_KEY = &CDATA_MT_KEY;
static const char cdatamt[] = ""
  "local ok, ffi = pcall(require, 'ffi')\n"
  "if ok and not jit then\n"
  "  return ffi.debug().cdata_mt\n"
  "else\n"
  "  return {}\n"
  "end\n";

static int luaT_iscdata(lua_State *L, int ud)
{
  int type = lua_type(L, ud);
  if(type == 10)
    return 1;
  if(type != LUA_TUSERDATA)
    return 0;
  if(!lua_getmetatable(L, ud))
    return 0;

  lua_pushlightuserdata(L, CDATA_MT_KEY);
  lua_rawget(L, LUA_REGISTRYINDEX);
  if (lua_isnil(L, -1))
  {
    // initialize cdata metatable
    lua_pop(L, 1);
    if(luaL_dostring(L, cdatamt))
      luaL_error(L, "internal error (could not load cdata mt): %s", lua_tostring(L, -1));

    lua_pushlightuserdata(L, CDATA_MT_KEY);
    lua_pushvalue(L, -2);
    lua_rawset(L, LUA_REGISTRYINDEX);
  }

  int iscdata = lua_rawequal(L, -1, -2);
  lua_pop(L, 2);
  return iscdata;
}

const char* luaT_typename(lua_State *L, int ud)
{
  if(luaT_iscdata(L, ud))
    return luaT_cdataname(L, ud, NULL);
  else if(lua_getmetatable(L, ud))
  {
    const char *tname = NULL;
    lua_rawget(L, LUA_REGISTRYINDEX);
    if(lua_isstring(L, -1))
      tname = lua_tostring(L, -1);
    lua_pop(L, 1); /* the string/nil */
    return tname;
  }
  return NULL;
}

void luaT_pushudata(lua_State *L, void *udata, const char *tname)
{
  if(udata)
  {
    void **udata_p = lua_newuserdata(L, sizeof(void*));
    *udata_p = udata;
    if(!luaT_pushmetatable(L, tname))
      luaL_error(L, "Torch internal problem: cannot find metatable for type <%s>", tname);
    lua_setmetatable(L, -2);
  }
  else
    lua_pushnil(L);
}

void *luaT_toudata(lua_State *L, int ud, const char *tname)
{
  void **p = lua_touserdata(L, ud);
  if(p != NULL) /* value is a userdata? */
  {
    if(!luaT_pushmetatable(L, tname))
      luaL_error(L, "Torch internal problem: cannot find metatable for type <%s>", tname);

    /* initialize the table we want to get the metatable on */
    /* note that we have to be careful with indices, as we just inserted stuff */
    lua_pushvalue(L, (ud < 0 ? ud - 1 : ud));
    while(lua_getmetatable(L, -1)) /* get the next metatable */
    {
      lua_remove(L, -2); /* remove the previous metatable [or object, if first time] */
      if(lua_rawequal(L, -1, -2))
      {
        lua_pop(L, 2);  /* remove the two metatables */
        return *p;
      }
    }
    lua_pop(L, 2); /* remove the two metatables */
  }
  return NULL;
}

int luaT_isudata(lua_State *L, int ud, const char *tname)
{
  if(luaT_toudata(L, ud, tname))
    return 1;
  else
    return 0;
}

void *luaT_checkudata(lua_State *L, int ud, const char *tname)
{
  void *p = luaT_toudata(L, ud, tname);
  if(!p)
    luaT_typerror(L, ud, tname);
  return p;
}

void luaT_pushlong(lua_State *L, long n)
{
#if LUA_VERSION_NUM >= 503
  /* Only push the value as an integer if it fits in lua_Integer,
   or if the lua_Number representation will be even worse */
  if (sizeof(lua_Integer) >= sizeof(long) || sizeof(lua_Number) <= sizeof(lua_Integer)) {
    lua_pushinteger(L, n);
  } else {
    lua_pushnumber(L, (lua_Number)n);
  }
#else
  lua_pushnumber(L, (lua_Number)n);
#endif
}

long luaT_checklong(lua_State *L, int idx)
{
#if LUA_VERSION_NUM >= 503
  if (sizeof(lua_Integer) >= sizeof(long) || sizeof(lua_Number) <= sizeof(lua_Integer)) {
    return (long)luaL_checkinteger(L, idx);
  } else {
    return (long)luaL_checknumber(L, idx);
  }
#else
  return (long)luaL_checknumber(L, idx);
#endif
}

long luaT_tolong(lua_State *L, int idx)
{
#if LUA_VERSION_NUM == 503
  if (sizeof(lua_Integer) >= sizeof(long) || sizeof(lua_Number) <= sizeof(lua_Integer)) {
    return (long)lua_tointeger(L, idx);
  } else {
    return (long)lua_tonumber(L, idx);
  }
#else
  return (long)lua_tonumber(L, idx);
#endif
}

void luaT_pushinteger(lua_State *L, ptrdiff_t n)
{
#if LUA_VERSION_NUM >= 503
  /* Only push the value as an integer if it fits in lua_Integer,
   or if the lua_Number representation will be even worse */
  if (sizeof(lua_Integer) >= sizeof(ptrdiff_t) || sizeof(lua_Number) <= sizeof(lua_Integer)) {
    lua_pushinteger(L, n);
  } else {
    lua_pushnumber(L, (lua_Number)n);
  }
#else
  lua_pushnumber(L, (lua_Number)n);
#endif
}

ptrdiff_t luaT_checkinteger(lua_State *L, int idx)
{
#if LUA_VERSION_NUM >= 503
  if (sizeof(lua_Integer) >= sizeof(ptrdiff_t) || sizeof(lua_Number) <= sizeof(lua_Integer)) {
    return (ptrdiff_t)luaL_checkinteger(L, idx);
  } else {
    return (ptrdiff_t)luaL_checknumber(L, idx);
  }
#else
  return (ptrdiff_t)luaL_checknumber(L, idx);
#endif
}

void *luaT_getfieldcheckudata(lua_State *L, int ud, const char *field, const char *tname)
{
  void *p;
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  p = luaT_toudata(L, -1, tname);
  if(!p)
    luaL_error(L, "bad argument #%d (field %s is not a %s)", ud, field, tname);
  return p;
}

void *luaT_getfieldchecklightudata(lua_State *L, int ud, const char *field)
{
  void *p;
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);

  if(!lua_islightuserdata(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a light userdata)", ud, field);

  p = lua_touserdata(L, -1);

  return p;
}

double luaT_getfieldchecknumber(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isnumber(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a number)", ud, field);
  return lua_tonumber(L, -1);
}

int luaT_getfieldcheckint(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isnumber(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a number)", ud, field);
  return (int)lua_tonumber(L, -1);
}

const char* luaT_getfieldcheckstring(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isstring(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a string)", ud, field);
  return lua_tostring(L, -1);
}

int luaT_getfieldcheckboolean(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isboolean(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a boolean)", ud, field);
  return lua_toboolean(L, -1);
}

void luaT_getfieldchecktable(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_istable(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a table)", ud, field);
}

/**** type checks as in luaL ****/
int luaT_typerror(lua_State *L, int ud, const char *tname)
{
  const char *msg;
  const char *tnameud = luaT_typename(L, ud);

  if(!tnameud)
    tnameud = lua_typename(L, ud);

  msg = lua_pushfstring(L, "%s expected, got %s",
                        tname,
                        (tnameud ? tnameud : "unknown object"));

  return luaL_argerror(L, ud, msg);
}

int luaT_checkboolean(lua_State *L, int ud)
{
  if(!lua_isboolean(L, ud))
    luaT_typerror(L, ud, lua_typename(L, LUA_TBOOLEAN));
  return lua_toboolean(L, ud);
}

int luaT_optboolean(lua_State *L, int ud, int def)
{
  if(lua_isnoneornil(L,ud))
    return def;

  return luaT_checkboolean(L, ud);
}

void luaT_registeratname(lua_State *L, const struct luaL_Reg *methods, const char *name)
{
  int idx = lua_gettop(L);

  luaL_checktype(L, idx, LUA_TTABLE);
  lua_pushstring(L, name);
  lua_rawget(L, idx);

  if(lua_isnil(L, -1))
  {
    lua_pop(L, 1);
    lua_pushstring(L, name);
    lua_newtable(L);
    lua_rawset(L, idx);

    lua_pushstring(L, name);
    lua_rawget(L, idx);
  }

  luaT_setfuncs(L, methods, 0);
  lua_pop(L, 1);
}


/* returns the name of the class itself (sans nesting) */
const char* luaT_classrootname(const char *tname)
{
  int idx;
  int sz = strlen(tname);

  for(idx = sz-1; idx >= 0 ; idx--)
  {
    if(tname[idx] == '.')
      return tname+idx+1;
  }
  return tname;
}

/* parent_name must be a buffer at least as big as tname.
 * If class has a parent, returns true; and, sets
 * parent name to that of full parent hierarchy (e.g.
 * given class `A.b.c`, sets parent_name to `A.b`)
 */
int luaT_fullparentname(const char *tname, char *parent_name)
{
  int sz = strlen(tname);
  int idx;
  for(idx = sz-1; idx > 0 ; idx--)
    if(tname[idx] == '.' || tname[idx] == '\0') break;

  if (idx > 0) strncpy(parent_name, tname, idx);
  parent_name[idx] = '\0';
  return tname[idx] == '.';
}

/* alias for ensuring backwards compatibilty;
 * use of luaT_fullparentname is preferred.
 */
int luaT_classmodulename(const char *tname, char *parent_name)
{
  return luaT_fullparentname(tname, parent_name);
}

/* parent_name must be a buffer at least as big as tname.
 * If class has a parent, returns true; and, sets
 * parent name to that of outermost parent (e.g.
 * given class `A.b.c`, sets parent_name to `A`)
 */
int luaT_outerparentname(const char *tname, char *parent_name)
{
  char chars[] = {'.', '\0'};
  size_t idx;
  idx = strcspn(tname, chars);
  strncpy(parent_name, tname, idx);
  parent_name[idx] = '\0';
  return tname[idx] == '.';
}

/* parent_name must be a buffer at least as big as tname.
 * If class has a parent, returns true; and, sets parent
 * name to that of innermost parent (e.g. given class
 * `A.b.c`, sets parent_name to `b`). In the comments
 * below, the inner parent name is abbreviated as IPN.
 */
int luaT_innerparentname(const char *tname, char *parent_name)
{
  int sz = strlen(tname);
  int tail, head;
  for(tail = sz-1; tail >= 0 ; tail--) // tail points to
    if(tname[tail] == '.') break;      // just past IPN

  if (tail == 0) return 0;

  for(head = tail-1; head >= 0; head--) // head points to
    if(tname[head] == '.') break;       // just before IPN

  head += 1; // update head to start of IPN
  tail -= head; // update tail to strlen(IPN)
  strncpy(parent_name, tname+head, tail);
  parent_name[tail] = '\0';
  return 1;
}

/* Method for pushing a class's immediate parent to the
 * stack (e.g. given class `A.b.c`, pushes `b` to the stack)
 */
void luaT_getinnerparent(lua_State *L, const char *tname)
{
  /* Local variables */
  char term[256];
  char chars[] = {'.', '\0'};
  const char *tname_full = tname; // used for error case

  /* Get outermost table from Lua */
  int n = strcspn(tname, chars);
  strncpy(term, tname, n);
  term[n] = '\0';
  lua_getglobal(L, term);
  tname  += n + 1;

  /* Traverse hierarchy down to last table*/
  n = strcspn(tname, chars);
  while(n < strlen(tname))
  {
    /* Check that current parent is a table (i.e. a module) */
    if(!lua_istable(L, -1)){
      strncpy(term, tname_full, tname - tname_full - 1);
      term[tname - tname_full] = '\0';
      luaL_error(L, "while creating metatable %s: bad argument #1 (%s is an invalid module name)", tname_full, term);
    }
    strncpy(term, tname, n);
    term[n] = '\0';
    lua_getfield(L, -1, term);
    lua_remove(L, -2);
    tname += n + 1;
    n = strcspn(tname, chars); // prepare for next
  }

  /* Check that resulting parent is a table (i.e. a module) */
  if(!lua_istable(L, -1)){
    strncpy(term, tname_full, tname - tname_full - 1);
    term[tname - tname_full] = '\0';
    luaL_error(L, "while creating metatable %s: bad argument #1 (%s is an invalid module name)", tname_full, term);
  }
}


int luaT_lua_newmetatable(lua_State *L)
{
  /* Local Variables */
  const char* tname = luaL_checkstring(L, 1);
  char parent_name[256];
  int is_in_module = 0;

  /* Argument Checking */
  lua_settop(L, 6);
  luaL_argcheck(L, lua_isnoneornil(L, 2) || lua_isstring(L, 2), 2, "parent class name or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 3) || lua_isfunction(L, 3), 3, "constructor function or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 4) || lua_isfunction(L, 4), 4, "destructor function or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 5) || lua_isfunction(L, 5), 5, "factory function or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 6) || lua_istable(L, 6), 6, "module table or nil expected");

  /* Push immediate parent module to stack */
  if(lua_isnoneornil(L, 6)) {
    lua_pop(L, 1); /* remove the nil */
    is_in_module = luaT_fullparentname(tname, parent_name);
    if (is_in_module)
      luaT_getinnerparent(L, tname);
    else
      lua_pushglobaltable(L);
  }

  if(!lua_istable(L, -1))
    luaL_error(L, "while creating metatable %s: bad argument #1 (%s is an invalid module name)", tname, parent_name);

  /* we first create the new metaclass if we have to */
  if(!luaT_pushmetatable(L, tname))
  {
    /* create the metatable */
    lua_newtable(L);

    /* registry[name] = metatable */
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_REGISTRYINDEX, tname);

    /* registry[metatable] = tname */
    lua_pushvalue(L, -1);
    lua_pushstring(L, tname);
    lua_rawset(L, LUA_REGISTRYINDEX);

    /* __index handling */
    lua_pushcfunction(L, luaT_mt__index);
    lua_setfield(L, -2, "__index");

    /* __newindex handling */
    lua_pushcfunction(L, luaT_mt__newindex);
    lua_setfield(L, -2, "__newindex");

    /* __typename contains the typename */
    lua_pushstring(L, tname);
    lua_setfield(L, -2, "__typename");

    /* __metatable is self */
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__metatable");

    /* by default, __version equals 1 */
    lua_pushnumber(L, 1);
    lua_setfield(L, -2, "__version");

    /* assign default operator functions */
    lua_pushcfunction(L, luaT_mt__tostring);
    lua_setfield(L, -2, "__tostring");

    lua_pushcfunction(L, luaT_mt__add);
    lua_setfield(L, -2, "__add");

    lua_pushcfunction(L, luaT_mt__sub);
    lua_setfield(L, -2, "__sub");

    lua_pushcfunction(L, luaT_mt__mul);
    lua_setfield(L, -2, "__mul");

    lua_pushcfunction(L, luaT_mt__div);
    lua_setfield(L, -2, "__div");

    lua_pushcfunction(L, luaT_mt__mod);
    lua_setfield(L, -2, "__mod");

    lua_pushcfunction(L, luaT_mt__pow);
    lua_setfield(L, -2, "__pow");

    lua_pushcfunction(L, luaT_mt__unm);
    lua_setfield(L, -2, "__unm");

    lua_pushcfunction(L, luaT_mt__concat);
    lua_setfield(L, -2, "__concat");

    lua_pushcfunction(L, luaT_mt__len);
    lua_setfield(L, -2, "__len");

    lua_pushcfunction(L, luaT_mt__eq);
    lua_setfield(L, -2, "__eq");

    lua_pushcfunction(L, luaT_mt__lt);
    lua_setfield(L, -2, "__lt");

    lua_pushcfunction(L, luaT_mt__le);
    lua_setfield(L, -2, "__le");

    lua_pushcfunction(L, luaT_mt__call);
    lua_setfield(L, -2, "__call");
  }

  /* we assign the parent class if necessary */
  if(!lua_isnoneornil(L, 2))
  {
    if(lua_getmetatable(L, -1))
      luaL_error(L, "class %s has been already assigned a parent class\n", tname);
    else
    {
      const char* parent_tname = luaL_checkstring(L, 2);
      if(!luaT_pushmetatable(L, parent_tname))
        luaL_error(L, "bad argument #2 (invalid parent class name %s)", parent_tname);
      lua_setmetatable(L, -2);
    }
  }

  /* register the destructor function  */
  if(!lua_isnoneornil(L, 4))
  {
    /* does it exists already? */
    lua_pushstring(L, "__gc");
    lua_rawget(L, -2);

    if(lua_isnil(L, -1))
    {
      lua_pop(L, 1); /* pop nil */
      lua_pushstring(L, "__gc");
      lua_pushvalue(L, 4);
      lua_rawset(L, -3);
    }
    else
      luaL_error(L, "%s has been already assigned a destructor", tname);
  }

  /* register the factory function  */
  if(!lua_isnoneornil(L, 5))
  {
    /* does it exists already? */
    lua_pushstring(L, "__factory");
    lua_rawget(L, -2);

    if(lua_isnil(L, -1))
    {
      lua_pop(L, 1); /* pop nil */
      lua_pushstring(L, "__factory");
      lua_pushvalue(L, 5);
      lua_rawset(L, -3);
    }
    else
      luaL_error(L, "%s has been already assigned a factory", tname);
  }

  /******** Constructor table and metatable ********/
  lua_pushstring(L, "__constructor");
  lua_rawget(L, -2);
  if(lua_isnil(L, -1))
  {
    lua_pop(L, 1);                        /* pop nil */
    lua_newtable(L);                      /* fancy table */
    lua_newtable(L);                      /* fancy metatable */

    lua_pushvalue(L, -3);                 /* metatable */
    lua_setfield(L, -2, "__index");       /* so we can get the methods */

    lua_pushcfunction(L, luaT_cmt__newindex);
    lua_setfield(L, -2, "__newindex");    /* so we add new methods */

    lua_pushcfunction(L, luaT_cmt__call);
    lua_setfield(L, -2, "__call");        /* so we can create, we are here for only that */

    lua_pushvalue(L, -3);
    lua_setfield(L, -2, "__metatable");   /* redirect to metatable with methods */

    lua_setmetatable(L, -2);              /* constructor metatable is ... this fancy metatable */

    /* set metatable[__constructor] = constructor-metatable */
    lua_pushstring(L, "__constructor");
    lua_pushvalue(L, -2);
    lua_rawset(L, -4);
  }

  /* register the constructor function  */
  if(!lua_isnoneornil(L, 3))
  {
    /* get constructor metatable */
    lua_getmetatable(L, -1);

    /* does it exists already? */
    lua_pushstring(L, "__new");
    lua_rawget(L, -2);

    if(lua_isnil(L, -1))
    {
      lua_pop(L, 1); /* pop nil */
      lua_pushstring(L, "__new");
      lua_pushvalue(L, 3);
      lua_rawset(L, -3);

      /* set "new" in the metatable too */
      lua_pushstring(L, "new");
      lua_pushvalue(L, 3);
      lua_rawset(L, -5);
    }
    else
      luaL_error(L, "%s has been already assigned a constructor", tname);

    /* pop constructor metatable */
    lua_pop(L, 1);
  }

  /* module.name = constructor metatable */
  lua_setfield(L, 6, luaT_classrootname(tname));

  return 1; /* returns the metatable */
}

/* Lua only utility functions */

/* add any custom type, provided the object has a metatable */
int luaT_lua_metatype(lua_State *L)
{
  if( (lua_gettop(L) != 2) && (lua_gettop(L) != 3) )
    luaL_error(L, "expecting: string table [ctype]");

  luaL_checkstring(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);

  if(lua_gettop(L) == 3)
  {
    if(!luaT_cdataname(L, 3, lua_tostring(L, 1)))
      luaL_error(L, "could not register cdata type -- missing ffi library?");
  }

  /* registry[name] = metatable */
  lua_pushvalue(L, 1);
  lua_pushvalue(L, 2);
  lua_rawset(L, LUA_REGISTRYINDEX);

  /* registry[metatable] = tname */
  lua_pushvalue(L, 2);
  lua_pushvalue(L, 1);
  lua_rawset(L, LUA_REGISTRYINDEX);

  return 0;
}

/* return a userdata from a C pointer */
/* you are better to know what you are doing */
int luaT_lua_pushudata(lua_State *L)
{
  void *udata = NULL;
  const char *tname = luaL_checkstring(L, 2);

  if(lua_type(L, 1) == 10)
    udata = *((void**)lua_topointer(L, 1));
  else if(luaT_iscdata(L, 1))
    udata = ((void**)lua_topointer(L, 1))[4];
  else if(lua_isnumber(L, 1))
    udata = (void*)(uintptr_t)lua_tonumber(L, 1);
  else
    luaL_argerror(L, 1, "expecting number or cdata");

  luaT_pushudata(L, udata, tname);

  return 1;
}

int luaT_lua_factory(lua_State *L)
{
  const char* tname = luaL_checkstring(L, 1);
  if(luaT_pushmetatable(L, tname) && !lua_isnil(L, -1))
  {
    lua_pushstring(L, "__factory");
    lua_rawget(L, -2);
  }
  else
  {
    lua_pushnil(L);
  }
  return 1;
}

int luaT_lua_getconstructortable(lua_State *L)
{
  const char* tname = luaL_checkstring(L, 1);
  if(luaT_pushmetatable(L, tname))
  {
    lua_pushstring(L, "__constructor");
    lua_rawget(L, -2);
    return 1;
  }
  return 0;
}


int luaT_lua_typename(lua_State *L)
{
  const char* tname = NULL;
  luaL_checkany(L, 1);
  if((tname = luaT_typename(L, 1)))
  {
    lua_pushstring(L, tname);
    return 1;
  }
  return 0;
}

int luaT_lua_isequal(lua_State *L)
{
  if(lua_isuserdata(L, 1) && lua_isuserdata(L, 2))
  {
    void **u1, **u2;
    luaL_argcheck(L, luaT_typename(L, 1), 1, "Torch object expected");
    luaL_argcheck(L, luaT_typename(L, 2), 2, "Torch object expected");

    u1 = lua_touserdata(L, 1);
    u2 = lua_touserdata(L, 2);
    if(*u1 == *u2)
      lua_pushboolean(L, 1);
    else
      lua_pushboolean(L, 0);
  }
  else if(lua_istable(L, 1) && lua_istable(L, 2))
    lua_pushboolean(L, lua_rawequal(L, 1, 2));
  else
    lua_pushboolean(L, 0);
  return 1;
}

static void luaT_pushpointer(lua_State *L, const void *ptr)
{
#if LUA_VERSION_NUM >= 503
  // this assumes that lua_Integer is a ptrdiff_t
  if (sizeof(void *) > sizeof(lua_Integer))
    luaL_error(L, "Pointer value can't be represented as a Lua integer (an overflow would occur)");
  lua_pushinteger(L, (uintptr_t)(ptr));
#else
  // 2^53 - this assumes that lua_Number is a double
  if ((uintptr_t)ptr > 9007199254740992LLU)
    luaL_error(L, "Pointer value can't be represented as a Lua number (an overflow would occur)");
  lua_pushnumber(L, (uintptr_t)(ptr));
#endif
}

int luaT_lua_pointer(lua_State *L)
{
  if(lua_type(L, 1) == 10) /* luajit cdata */
  {
    /* we want the pointer holded by cdata */
    /* not the pointer on the cdata object */
    const void* ptr = *((void**)lua_topointer(L, 1));
    luaT_pushpointer(L, ptr);
    return 1;
  }
  else if (luaT_iscdata(L, 1)) /* luaffi cdata */
  {
    void** ptr = (void**)lua_touserdata(L, 1);
    luaT_pushpointer(L, ptr[4]);
    return 1;
  }
  else if(lua_isuserdata(L, 1))
  {
    void **ptr;
    luaL_argcheck(L, luaT_typename(L, 1), 1, "Torch object expected");
    ptr = lua_touserdata(L, 1);
    luaT_pushpointer(L, *ptr);
    return 1;
  }
  else if(lua_istable(L, 1) || lua_isthread(L, 1) || lua_isfunction(L, 1))
  {
    const void* ptr = lua_topointer(L, 1);
    luaT_pushpointer(L, ptr);
    return 1;
  }
  else if(lua_isstring(L, 1))
  {
    const char* ptr = lua_tostring(L, 1);
    luaT_pushpointer(L, ptr);
    return 1;
  }
  else
    luaL_error(L, "Torch object, table, thread, cdata or function expected");

  return 0;
}

int luaT_lua_setenv(lua_State *L)
{
  if(!lua_isfunction(L, 1) && !lua_isuserdata(L, 1))
    luaL_typerror(L, 1, "function or userdata");
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_setuservalue(L, 1);
  return 0;
}

int luaT_lua_getenv(lua_State *L)
{
  if(!lua_isfunction(L, 1) && !lua_isuserdata(L, 1))
    luaL_typerror(L, 1, "function or userdata");
  lua_getuservalue(L, 1);
  if (lua_isnil(L, -1))
    lua_newtable(L);
  return 1;
}

int luaT_lua_getmetatable(lua_State *L)
{
  const char *tname = luaL_checkstring(L, 1);
  if(luaT_pushmetatable(L, tname))
    return 1;
  return 0;
}

int luaT_lua_version(lua_State *L)
{
  luaL_checkany(L, 1);

  if(luaT_iscdata(L, 1))
  {
    const char *tname = luaT_cdataname(L, 1, NULL);
    if(tname)
    {
      luaT_pushmetatable(L, tname);
      lua_pushstring(L, "__version");
      lua_rawget(L, -2);
      return 1;
    }
    return 0;
  }
  else if(lua_getmetatable(L, 1))
  {
    lua_pushstring(L, "__version");
    lua_rawget(L, -2);
    return 1;
  }
  return 0;
}

int luaT_lua_setmetatable(lua_State *L)
{
  const char *tname = luaL_checkstring(L, 2);
  luaL_checktype(L, 1, LUA_TTABLE);

  if(!luaT_pushmetatable(L, tname))
    luaL_error(L, "unknown typename %s\n", tname);
  lua_setmetatable(L, 1);

  return 1;
}

/* metatable operator methods */
static int luaT_mt__index(lua_State *L)
{
  if(!lua_getmetatable(L, 1))
    luaL_error(L, "critical internal indexing error: no metatable found");

  if(!lua_istable(L, -1))
    luaL_error(L, "critical internal indexing error: not a metatable");

  /* test for __index__ method first */
  lua_getfield(L, -1, "__index__");
  if(!lua_isnil(L, -1))
  {
    int result;

    if(!lua_isfunction(L, -1))
      luaL_error(L, "critical internal indexing error: __index__ is not a function");

    lua_pushvalue(L, 1);
    lua_pushvalue(L, 2);

    lua_call(L, 2, LUA_MULTRET); /* DEBUG: risque: faut vraiment retourner 1 ou 2 valeurs... */

    result = lua_toboolean(L, -1);
    lua_pop(L, 1);

    if(result)
      return 1;

    /* on the stack: 1. the object 2. the value 3. the metatable */
    /* apparently, __index wants only one element returned */
    /* return lua_gettop(L)-3; */

  }
  else
    lua_pop(L, 1); /* remove nil __index__ on the stack */

  lua_pushvalue(L, 2);
  lua_gettable(L, -2);

  return 1;
}

static int luaT_mt__newindex(lua_State *L)
{
  if(!lua_getmetatable(L, 1))
    luaL_error(L, "critical internal indexing error: no metatable found");

  if(!lua_istable(L, -1))
    luaL_error(L, "critical internal indexing error: not a metatable");

  /* test for __newindex__ method first */
  lua_getfield(L, -1, "__newindex__");
  if(!lua_isnil(L, -1))
  {
    int result;

    if(!lua_isfunction(L, -1))
      luaL_error(L, "critical internal indexing error: __newindex__ is not a function");

    lua_pushvalue(L, 1);
    lua_pushvalue(L, 2);
    lua_pushvalue(L, 3);

    lua_call(L, 3, 1); /* DEBUG: risque: faut vraiment retourner qqch */

    result = lua_toboolean(L, -1);
    lua_pop(L, 1);

    if(result)
      return 0;
  }
  else
    lua_pop(L, 1); /* remove nil __newindex__ on the stack */

  lua_pop(L, 1);    /* pop the metatable */
  if(lua_istable(L, 1))
    lua_rawset(L, 1);
  else
    luaL_error(L, "the class %s cannot be indexed", luaT_typename(L, 1));

  return 0;
}


#define MT_UNI_OPERATOR_GET_HANDLER(NAME)                               \
    if(!lua_getmetatable(L, 1))                                         \
      luaL_error(L, "internal error in __" #NAME ": no metatable");

#define MT_BIN_OPERATOR_GET_HANDLER(NAME)                               \
    if(!lua_getmetatable(L, 1) && !lua_getmetatable(L,2) )              \
      luaL_error(L, "internal error in __" #NAME                        \
              ": no metatable in both operands");

#define MT_DECLARE_OPERATOR_BODY(NAME, NIL_BEHAVIOR)                    \
                                                                        \
    lua_getfield(L, -1, "__" #NAME "__");                               \
    if(lua_isnil(L, -1))                                                \
    {                                                                   \
      NIL_BEHAVIOR;                                                     \
    }                                                                   \
    else                                                                \
    {                                                                   \
      if(lua_isfunction(L, -1))                                         \
      {                                                                 \
        lua_insert(L, 1); /* insert function */                         \
        lua_pop(L, 1); /* remove metatable */                           \
        lua_call(L, lua_gettop(L)-1, LUA_MULTRET);                      \
          /* we return the result of the call */                        \
        return lua_gettop(L);                                           \
      }                                                                 \
      /* we return the thing the user left in __tostring__ */           \
    }                                                                   \
    return 0;                                                           \

/* note: check dans metatable pour ca, donc necessaire */
#define MT_DECLARE_OPERATOR(NAME, NIL_BEHAVIOR)                         \
  int luaT_mt__##NAME(lua_State *L)                                     \
  {                                                                     \
    MT_UNI_OPERATOR_GET_HANDLER(NAME)                                   \
    MT_DECLARE_OPERATOR_BODY(NAME,NIL_BEHAVIOR)                         \
  }

#define MT_DECLARE_BIN_OPERATOR(NAME, NIL_BEHAVIOR)                     \
  int luaT_mt__##NAME(lua_State *L)                                     \
  {                                                                     \
    MT_BIN_OPERATOR_GET_HANDLER(NAME)                                   \
    MT_DECLARE_OPERATOR_BODY(NAME,NIL_BEHAVIOR)                         \
  }


#define BIN_OPERATOR_ERROR(NAME)                                        \
    luaL_error(L, "both %s and %s have no " #NAME " operator",          \
            luaT_typename(L, 1), luaT_typename(L,2))

MT_DECLARE_BIN_OPERATOR(add,    BIN_OPERATOR_ERROR(addition) )
MT_DECLARE_BIN_OPERATOR(sub,    BIN_OPERATOR_ERROR(substraction) )
MT_DECLARE_BIN_OPERATOR(mul,    BIN_OPERATOR_ERROR(multiplication) )
MT_DECLARE_BIN_OPERATOR(div,    BIN_OPERATOR_ERROR(division) )
MT_DECLARE_BIN_OPERATOR(mod,    BIN_OPERATOR_ERROR(modulo) )
MT_DECLARE_BIN_OPERATOR(pow,    BIN_OPERATOR_ERROR(power) )
MT_DECLARE_BIN_OPERATOR(concat, BIN_OPERATOR_ERROR(concat) )
MT_DECLARE_BIN_OPERATOR(eq,
                    lua_settop(L, 2);
                    lua_pushcfunction(L, luaT_lua_isequal);
                    lua_insert(L, 1);
                    lua_call(L, 2, 1);
                    return 1;)
MT_DECLARE_BIN_OPERATOR(lt, BIN_OPERATOR_ERROR(less-than) )
MT_DECLARE_BIN_OPERATOR(le, BIN_OPERATOR_ERROR(less-equal) )

MT_DECLARE_OPERATOR(tostring,
                    lua_pushstring(L, luaT_typename(L, 1));
                    return 1;)
MT_DECLARE_OPERATOR(call, luaL_error(L, "%s has no call operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(unm, luaL_error(L, "%s has no negation operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(len, luaL_error(L, "%s has no length operator", luaT_typename(L, 1)))


/* constructor metatable methods */
int luaT_cmt__call(lua_State *L)
{
  if(!lua_istable(L, 1))
    luaL_error(L, "internal error in __call: not a constructor table");

  if(!lua_getmetatable(L, 1))
    luaL_error(L, "internal error in __call: no metatable available");

  lua_pushstring(L, "__new");
  lua_rawget(L, -2);

  if(lua_isnil(L, -1))
    luaL_error(L, "no constructor available");

  lua_remove(L, 1); /* remove constructor atable */
  lua_insert(L, 1); /* insert constructor */
  lua_pop(L, 1);    /* remove fancy metatable */

  lua_call(L, lua_gettop(L)-1, LUA_MULTRET);
  return lua_gettop(L);
}

int luaT_cmt__newindex(lua_State *L)
{
  if(!lua_istable(L, 1))
    luaL_error(L, "internal error in __newindex: not a constructor table");

  if(!lua_getmetatable(L, 1))
    luaL_error(L, "internal error in __newindex: no metatable available");

  lua_pushstring(L, "__metatable");
  lua_rawget(L, -2);

  if(!lua_istable(L, -1))
    luaL_error(L, "internal error in __newindex: no metaclass available");

  lua_insert(L, 2);
  lua_pop(L, 1); /* remove the metatable over the constructor table */

  lua_rawset(L, -3);

  return 0;
}

/******************** deprecated functions ********************/
int luaT_pushmetaclass(lua_State *L, const char *tname)
{
  return luaT_pushmetatable(L, tname);
}

const char* luaT_id(lua_State *L, int ud)
{
  return luaT_typename(L, ud);
}

const char* luaT_id2typename(lua_State *L, const char *id)
{
  return id;
}

const char* luaT_typename2id(lua_State *L, const char *tname)
{
  return luaT_typenameid(L, tname);
}

int luaT_getmetaclass(lua_State *L, int index)
{
  return lua_getmetatable(L, index);
}

const char* luaT_checktypename2id(lua_State *L, const char *tname)
{
  const char* id = luaT_typenameid(L, tname);
  if(!id)
    luaL_error(L, "unknown class <%s>", tname);
  return id;
}

void luaT_registeratid(lua_State *L, const struct luaL_Reg *methods, const char *id)
{
  luaT_registeratname(L, methods, id);
}

/**************************************************************/
