#include "general.h"

#ifdef _WIN32

#include <windows.h>
#include <assert.h>
#define TimeType __int64
static __declspec( thread ) TimeType ticksPerSecond = 0;

/*
 * There is an example of getrusage for windows in following link:
 * https://github.com/openvswitch/ovs/blob/master/lib/getrusage-windows.c
 */

#else

#include <sys/time.h>
#include <sys/resource.h>
#define TimeType double

#endif

typedef struct _Timer
{
    int isRunning;

    TimeType totalrealtime;
    TimeType totalusertime;
    TimeType totalsystime;

    TimeType startrealtime;
    TimeType startusertime;
    TimeType startsystime;
} Timer;

static TimeType torch_Timer_realtime()
{
#ifdef _WIN32
  TimeType current;
  QueryPerformanceCounter(&current);
  return current;
#else
  struct timeval current;
  gettimeofday(&current, NULL);
  return (current.tv_sec + current.tv_usec/1000000.0);
#endif
}

static TimeType torch_Timer_usertime()
{
#ifdef _WIN32
  return torch_Timer_realtime();
#else
  struct rusage current;
  getrusage(RUSAGE_SELF, &current);
  return (current.ru_utime.tv_sec + current.ru_utime.tv_usec/1000000.0);
#endif
}

static TimeType torch_Timer_systime()
{
#ifdef _WIN32
  return 0;
#else
  struct rusage current;
  getrusage(RUSAGE_SELF, &current);
  return (current.ru_stime.tv_sec + current.ru_stime.tv_usec/1000000.0);
#endif
}

static int torch_Timer_new(lua_State *L)
{
#ifdef _WIN32
  if (ticksPerSecond == 0)
  {
    assert(sizeof(LARGE_INTEGER) == sizeof(__int64));
    QueryPerformanceFrequency(&ticksPerSecond);
  }
#endif
  Timer *timer = luaT_alloc(L, sizeof(Timer));
  timer->isRunning = 1;
  timer->totalrealtime = 0;
  timer->totalusertime = 0;
  timer->totalsystime = 0;
  timer->startrealtime = torch_Timer_realtime();
  timer->startusertime = torch_Timer_usertime();
  timer->startsystime = torch_Timer_systime();
  luaT_pushudata(L, timer, "torch.Timer");
  return 1;
}

static int torch_Timer_reset(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  timer->totalrealtime = 0;
  timer->totalusertime = 0;
  timer->totalsystime = 0;
  timer->startrealtime = torch_Timer_realtime();
  timer->startusertime = torch_Timer_usertime();
  timer->startsystime = torch_Timer_systime();
  lua_settop(L, 1);
  return 1;
}

static int torch_Timer_free(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  luaT_free(L, timer);
  return 0;
}

static int torch_Timer_stop(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  if(timer->isRunning)  
  {
    TimeType realtime = torch_Timer_realtime() - timer->startrealtime;
    TimeType usertime = torch_Timer_usertime() - timer->startusertime;
    TimeType systime = torch_Timer_systime() - timer->startsystime;
    timer->totalrealtime += realtime;
    timer->totalusertime += usertime;
    timer->totalsystime += systime;
    timer->isRunning = 0;
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_resume(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  if(!timer->isRunning)
  {
    timer->isRunning = 1;
    timer->startrealtime = torch_Timer_realtime();
    timer->startusertime = torch_Timer_usertime();
    timer->startsystime = torch_Timer_systime();
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_time(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  double realtime = (timer->isRunning ? (timer->totalrealtime + torch_Timer_realtime() - timer->startrealtime) : timer->totalrealtime);
  double usertime = (timer->isRunning ? (timer->totalusertime + torch_Timer_usertime() - timer->startusertime) : timer->totalusertime);
  double systime = (timer->isRunning ? (timer->totalsystime + torch_Timer_systime() - timer->startsystime) : timer->totalsystime);
#ifdef _WIN32
  realtime /= ticksPerSecond;
  usertime /= ticksPerSecond;
  systime  /= ticksPerSecond;
#endif
  lua_createtable(L, 0, 3);
  lua_pushnumber(L, realtime);
  lua_setfield(L, -2, "real");
  lua_pushnumber(L, usertime);
  lua_setfield(L, -2, "user");
  lua_pushnumber(L, systime);
  lua_setfield(L, -2, "sys");
  return 1;
}

static int torch_Timer___tostring__(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  lua_pushfstring(L, "torch.Timer [status: %s]", (timer->isRunning ? "running" : "stopped"));
  return 1;
}

static const struct luaL_Reg torch_Timer__ [] = {
  {"reset", torch_Timer_reset},
  {"stop", torch_Timer_stop},
  {"resume", torch_Timer_resume},
  {"time", torch_Timer_time},
  {"__tostring__", torch_Timer___tostring__},
  {NULL, NULL}
};

void torch_Timer_init(lua_State *L)
{
  luaT_newmetatable(L, "torch.Timer", NULL, torch_Timer_new, torch_Timer_free, NULL);
  luaT_setfuncs(L, torch_Timer__, 0);
  lua_pop(L, 1);
}
