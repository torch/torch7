#ifndef LUAT_UTILS_INC
#define LUAT_UTILS_INC

#ifdef __cplusplus
extern "C" {
#endif
#include <lua.h>
#include <lauxlib.h>
#ifdef __cplusplus
}
#endif

#ifndef LUA_EXTERNC
# ifdef __cplusplus
#  define LUA_EXTERNC extern "C"
# else
#  define LUA_EXTERNC extern
# endif
#endif

#if (defined(_MSC_VER) || defined(__MINGW32__))
# define DLL_EXPORT __declspec(dllexport)
# define DLL_IMPORT __declspec(dllimport)
# ifdef luaT_EXPORTS
#  define LUAT_API LUA_EXTERNC DLL_EXPORT
# else
#  define LUAT_API LUA_EXTERNC DLL_IMPORT
# endif
#else
# define DLL_EXPORT
# define DLL_IMPORT
# define LUAT_API LUA_EXTERNC
#endif

#if LUA_VERSION_NUM == 501
# define lua_pushglobaltable(L) lua_pushvalue(L, LUA_GLOBALSINDEX)
# define lua_setuservalue lua_setfenv
# define lua_getuservalue lua_getfenv
#else
# define lua_objlen lua_rawlen
static int luaL_typerror(lua_State *L, int narg, const char *tname)
{
  return luaL_error(L, "%s expected, got %s", tname, luaL_typename(L, narg));
}
#endif


/* C functions */

LUAT_API void* luaT_alloc(lua_State *L, ptrdiff_t size);
LUAT_API void* luaT_realloc(lua_State *L, void *ptr, ptrdiff_t size);
LUAT_API void luaT_free(lua_State *L, void *ptr);

LUAT_API void luaT_setfuncs(lua_State *L, const luaL_Reg *l, int nup);

LUAT_API const char* luaT_newlocalmetatable(lua_State *L, const char *tname, const char *parent_tname,
                                            lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory, int moduleidx);

LUAT_API const char* luaT_newmetatable(lua_State *L, const char *tname, const char *parenttname,
                                       lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory);

LUAT_API int luaT_pushmetatable(lua_State *L, const char *tname);

LUAT_API const char* luaT_typenameid(lua_State *L, const char *tname);
LUAT_API const char* luaT_typename(lua_State *L, int ud);

LUAT_API void luaT_pushudata(lua_State *L, void *udata, const char *tname);
LUAT_API void *luaT_toudata(lua_State *L, int ud, const char *tname);
LUAT_API int luaT_isudata(lua_State *L, int ud, const char *tname);
LUAT_API void *luaT_checkudata(lua_State *L, int ud, const char *tname);

LUAT_API void luaT_pushlong(lua_State *L, long n);
LUAT_API long luaT_checklong(lua_State *L, int idx);
LUAT_API long luaT_tolong(lua_State *L, int idx);

LUAT_API void luaT_pushinteger(lua_State *L, ptrdiff_t n);
LUAT_API ptrdiff_t luaT_checkinteger(lua_State *L, int idx);

LUAT_API void *luaT_getfieldcheckudata(lua_State *L, int ud, const char *field, const char *tname);
LUAT_API void *luaT_getfieldchecklightudata(lua_State *L, int ud, const char *field);
LUAT_API double luaT_getfieldchecknumber(lua_State *L, int ud, const char *field);
LUAT_API int luaT_getfieldcheckint(lua_State *L, int ud, const char *field);
LUAT_API const char* luaT_getfieldcheckstring(lua_State *L, int ud, const char *field);
LUAT_API int luaT_getfieldcheckboolean(lua_State *L, int ud, const char *field);
LUAT_API void luaT_getfieldchecktable(lua_State *L, int ud, const char *field);

LUAT_API int luaT_typerror(lua_State *L, int ud, const char *tname);

LUAT_API int luaT_checkboolean(lua_State *L, int ud);
LUAT_API int luaT_optboolean(lua_State *L, int ud, int def);

LUAT_API void luaT_registeratname(lua_State *L, const struct luaL_Reg *methods, const char *name);

/* utility functions */
LUAT_API const char *luaT_classrootname(const char *tname);
LUAT_API int luaT_classmodulename(const char *tname, char *module_name);

/* debug */
LUAT_API void luaT_stackdump(lua_State *L);

/* Lua functions */
LUAT_API int luaT_lua_newmetatable(lua_State *L);
LUAT_API int luaT_lua_factory(lua_State *L);
LUAT_API int luaT_lua_getconstructortable(lua_State *L);
LUAT_API int luaT_lua_typename(lua_State *L);
LUAT_API int luaT_lua_isequal(lua_State *L);
LUAT_API int luaT_lua_pointer(lua_State *L);
LUAT_API int luaT_lua_setenv(lua_State *L);
LUAT_API int luaT_lua_getenv(lua_State *L);
LUAT_API int luaT_lua_getmetatable(lua_State *L);
LUAT_API int luaT_lua_version(lua_State *L);
LUAT_API int luaT_lua_setmetatable(lua_State *L);
LUAT_API int luaT_lua_metatype(lua_State *L);
LUAT_API int luaT_lua_pushudata(lua_State *L);

/* deprecated functions */
/* ids have been replaced by string names to identify classes */
/* comments show what function (that you should use) they call now */
#if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define LUAT_DEPRECATED  __attribute__((__deprecated__))
#elif (defined(_MSC_VER) || defined(__MINGW32__))
#define LUAT_DEPRECATED __declspec(deprecated)
#else
#define LUAT_DEPRECATED
#endif

LUAT_API LUAT_DEPRECATED int luaT_pushmetaclass(lua_State *L, const char *tname); /* same as luaT_pushmetatable */
LUAT_API LUAT_DEPRECATED const char* luaT_id(lua_State *L, int ud); /* same as luaT_typename */
LUAT_API LUAT_DEPRECATED const char* luaT_id2typename(lua_State *L, const char *id); /*  same as luaT_typenameid */
LUAT_API LUAT_DEPRECATED const char* luaT_typename2id(lua_State *L, const char*); /* same as luaT_typenameid */
LUAT_API LUAT_DEPRECATED int luaT_getmetaclass(lua_State *L, int index); /* same as luaT_getmetatable */
LUAT_API LUAT_DEPRECATED const char* luaT_checktypename2id(lua_State *L, const char *tname);  /* same as luaT_typenameid */
LUAT_API LUAT_DEPRECATED void luaT_registeratid(lua_State *L, const struct luaL_Reg *methods, const char *id); /* same as luaT_registeratname */

#endif
