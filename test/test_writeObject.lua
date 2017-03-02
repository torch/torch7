require 'torch'

local myTester = torch.Tester()

local tests = torch.TestSuite()

function torch.HalfTensor:norm()
   return self:real():norm()
end

-- checks that an object can be written and unwritten
-- returns false if an error occurs
local function serializeAndDeserialize(obj)
   local file = torch.MemoryFile()
   file:binary()
   local ok, msg = pcall (file.writeObject, file, obj)
   myTester:assert(ok, 'error in writing an object'  )
   file:seek(1)
   local ok, copy = pcall(file.readObject, file)
   if not ok then print(copy) end
   myTester:assert(ok, 'error in reading an object ')
   return copy
end

function tests.test_can_write_a_nil_closure()
  local a
  local function closure()
    if not a then return 1 end
    return 0
  end

  local copyClosure = serializeAndDeserialize(closure)
  myTester:assert(copyClosure() == closure(), 'the closures should give same output')
end

function tests.test_nil_upvalues_in_closure()
  local a = 1
  local b
  local c = 2
  local function closure()
    if not b then return c end
    return a
  end

  local copyClosure = serializeAndDeserialize(closure)
  myTester:assert(copyClosure() == closure(), 'the closures should give same output')
end

function tests.test_global_function_in_closure()
  local x = "5"
  local function closure(str)
    return tonumber(str .. x)
  end

  local copyClosure = serializeAndDeserialize(closure)
  myTester:assert(copyClosure("3") == closure("3"), 'the closures should give same output')
end

function tests.test_a_recursive_closure()
  local foo

  foo = function (level)
    if level == 1 then return 1 end
    return 1+foo(level-1)
  end

  local copyFoo = serializeAndDeserialize(foo)
  myTester:assert(copyFoo(42) == foo(42), 'the closures should give same output')
end

function tests.test_a_tensor()
   for k,v in ipairs({"real", "half"}) do
      tests_test_a_tensor(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function tests_test_a_tensor(func)
   local x = func(torch.rand(5, 10))
   local xcopy = serializeAndDeserialize(x)
   myTester:assert(x:norm() == xcopy:norm(), 'tensors should be the same')
end

-- Regression test for bug reported in issue 456.
function tests.test_empty_table()
   local file = torch.MemoryFile()
   file:writeObject({})
end

function tests.test_error_msg()
   local torch = torch
   local inner = {
       baz = function(a) torch.somefunc() end
   }
   local outer = {
       theinner = inner
   }
   local function evil_func()
      outer.prop = 1
      image.compress(1)
   end
   local ok, msg = pcall(torch.save, 'saved.t7', evil_func)
   myTester:assert(not ok)
   myTester:assert(msg:find('at <%?>%.outer%.theinner%.baz%.torch') ~= nil)
end

function tests.test_warning_msg()
  local foo = {}
  torch.class('Bar', foo)

  local obj = foo.Bar()
  local tensor = torch.Tensor()
  obj.data = tensor:cdata() -- pick something NOT writable

  local file = torch.MemoryFile('rw'):binary()
  local ok, _ = pcall(torch.File.writeObject, file, obj)
  -- only a warning is printed on STDOUT:
  --   $ Warning: cannot write object field <data> of <Bar> <?>
  myTester:assert(ok)
  file:close()
end

function tests.test_referenced()
   local file = torch.MemoryFile('rw'):binary()
   file:referenced(false)

   local foo = 'bar'
   file:writeObject(foo)
   file:close()
end

function tests.test_shared_upvalues()
  if debug.upvalueid then
     local i=1
     local j=2

     local func = {}

     func.increment = function()
        i=i+1
        j=j+2
     end
     func.get_i = function()
        return i
     end
     func.get_j = function()
        return j
     end

     local copyFunc = serializeAndDeserialize(func)
     myTester:assert(copyFunc.get_i()==1)
     myTester:assert(copyFunc.get_j()==2)
     copyFunc.increment()
     myTester:assert(copyFunc.get_i()==2)
     myTester:assert(copyFunc.get_j()==4)
  else
     print('Not running shared upvalues test, as we are in Lua-5.1')
  end
end


-- checks that the hook function works properly
-- returns false if an error occurs
function tests.test_SerializationHook()
   -- Simpel uuid implementation from [https://gist.github.com/jrus/3197011]
   -- The only goal is to aoid collisions within the scope of tests,
   -- so more than enough.
   local random = math.random
   local function uuid()
       local template ='xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'
       return string.gsub(template, '[xy]', function (c)
           local v = (c == 'x') and random(0, 0xf) or random(8, 0xb)
           return string.format('%x', v)
       end)
   end
   local unique1 = uuid()
   local unique2 = uuid()
   local class = {}
   -- Create 2 classes
   local spec = torch.class('class.'.. unique1, class)
   function spec:test()
      return false
   end
   local gen = torch.class('class.' .. unique2, class)
   function gen:test()
      return true
   end
   local hook = function(object)
      local class = class
      local newObject = object
      if torch.typename(object) == 'class.'..unique1 then
         newObject = class[unique2]()
      end
      return newObject
   end

   -- Write to 2 files, first without hooking,
   -- second with hooking
   local file = torch.MemoryFile('rw')
   file:binary()
   local file2 = torch.MemoryFile('rw')
   file2:binary()
   local s = class[unique1]()
   local object = {s1 = s, v = 'test', g = class[unique2](), s2 = s}
   file:writeObject(object)
   file2:writeObject(object, nil, hook)

   -- unregister class[unique1] and try to reload the first serialized object
   if debug and debug.getregistry then
      local ok, res = pcall(function() classTestSerializationHook1 = nil debug.getregistry()[classTestSerializationHook1] = nil file:seek(1) return file:readObject() end)
      myTester:assert(not ok)
   else
      print('Not running serialization hook failure test because debug is missing.')
   end

   -- Try to reload the second serialized object
   local ok, clone = pcall(function() file2:seek(1) return file2:readObject()  end)

   -- Test that everything happened smoothly
   myTester:assert(clone.v == 'test')
   myTester:assert(torch.typename(clone.s1) == 'class.' .. unique2)
   myTester:assert(clone.s1:test() and clone.s2:test())
   myTester:assert(string.format('%x',torch.pointer(clone.s1)) == string.format('%x',torch.pointer(clone.s2)))
end

function tests.test_serializeToStorage()
   torch.save("foo.t7", "foo")
   local f = io.open("foo.t7", "rb")
   local size = f:seek("end")
   f:close()
   myTester:eq(
      torch.serializeToStorage("foo"):size(), size,
      "memory and disk serializations should have the same size"
   )
end

myTester:add(tests)
myTester:run()
if myTester.errors[1] then os.exit(1) end
