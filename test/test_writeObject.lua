local myTester = torch.Tester()

local tests = {}


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
   local x = torch.rand(5, 10)
   local xcopy = serializeAndDeserialize(x)
   myTester:assert(x:norm() == xcopy:norm(), 'tensors should be the same')
end

myTester:add(tests)
myTester:run()
