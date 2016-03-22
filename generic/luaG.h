#if !defined(real) || !defined(TH_GENERIC_FILE)
#error "luaG.h must not be included outside of a generic file."
#endif

#ifndef luaG_
#define luaG_(NAME) TH_CONCAT_3(luaG_,Real,NAME)
#endif

static void luaG_(pushreal)(lua_State *L, accreal n) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || LUA_VERSION_NUM < 503
	lua_pushnumber(L, (lua_Number)n);
#elif defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
	lua_pushinteger(L, (lua_Integer)n);
#else
	#error "unhandled real type in luaG_pushreal"
#endif
}

static real luaG_(checkreal)(lua_State *L, int idx) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || LUA_VERSION_NUM < 503
	return (lua_Number)luaL_checknumber(L, idx);
#elif defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
	return (lua_Integer)luaL_checkinteger(L, idx);
#else
	#error "unhandled real type in luaG_checkreal"
#endif
}

static real luaG_(optreal)(lua_State *L, int idx, real n) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || LUA_VERSION_NUM < 503
	return (lua_Number)luaL_optnumber(L, idx, (lua_Number)n);
#elif defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
	return (lua_Integer)luaL_optinteger(L, idx, (lua_Integer)n);
#else
	#error "unhandled real type in luaG_checkreal"
#endif
}
