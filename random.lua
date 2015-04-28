local wrap = require 'cwrap'

require 'torchcwrap'

local interface = wrap.CInterface.new()

interface:print(
   [[
#include "luaT.h"
#include "TH.h"

extern void torch_Generator_init(lua_State *L);
extern void torch_Generator_new(lua_State *L);
   ]])

for _,name in ipairs({"seed", "initialSeed"}) do
   interface:wrap(name,
                  string.format("THRandom_%s",name),
                  {{name='Generator', default=true},
                   {name="long", creturned=true}})
end

interface:wrap('manualSeed',
               'THRandom_manualSeed',
               {{name='Generator', default=true},
                {name="long"}})

interface:wrap('getRNGState',
                'THByteTensor_getRNGState',
                {{name='Generator', default=true},
                 {name='ByteTensor',default=true,returned=true,method={default='nil'}}
                 })

interface:wrap('setRNGState',
                'THByteTensor_setRNGState',
                {{name='Generator', default=true},
                 {name='ByteTensor',default=true,returned=true,method={default='nil'}}
                 })

interface:register("random__")
                
interface:print(
   [[
void torch_random_init(lua_State *L)
{
  torch_Generator_init(L);
  torch_Generator_new(L);
  lua_setfield(L, -2, "_gen");
  luaT_setfuncs(L, random__, 0);
}
]])

interface:tofile(arg[1])
