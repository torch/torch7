local mytester
local torchtest = torch.TestSuite()
local msize = 100
local precision

-- Lua 5.2 compatibility
local loadstring = loadstring or load
local unpack = unpack or table.unpack

local function maxdiff(x,y)
   local d = x-y
   if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.FloatTensor' or x:type() == 'torch.HalfTensor'  then
      return d:abs():max()
   else
      local dd = torch.Tensor():resize(d:size()):copy(d)
      return dd:abs():max()
   end
end


function torchtest.elementSize()
  local byte   =   torch.ByteStorage():elementSize()
  local char   =   torch.CharStorage():elementSize()
  local short  =  torch.ShortStorage():elementSize()
  local int    =    torch.IntStorage():elementSize()
  local long   =   torch.LongStorage():elementSize()
  local float  =  torch.FloatStorage():elementSize()
  local double = torch.DoubleStorage():elementSize()
  local half = torch.HalfStorage():elementSize()

  mytester:asserteq(byte,   torch.ByteTensor():elementSize())
  mytester:asserteq(char,   torch.CharTensor():elementSize())
  mytester:asserteq(short,  torch.ShortTensor():elementSize())
  mytester:asserteq(int,    torch.IntTensor():elementSize())
  mytester:asserteq(long,   torch.LongTensor():elementSize())
  mytester:asserteq(float,  torch.FloatTensor():elementSize())
  mytester:asserteq(double, torch.DoubleTensor():elementSize())
  mytester:asserteq(half, torch.HalfTensor():elementSize())

  mytester:assertne(byte, 0)
  mytester:assertne(char, 0)
  mytester:assertne(short, 0)
  mytester:assertne(int, 0)
  mytester:assertne(long, 0)
  mytester:assertne(float, 0)
  mytester:assertne(double, 0)
  mytester:assertne(half, 0)

  -- These tests are portable, not necessarily strict for your system.
  mytester:asserteq(byte, 1)
  mytester:asserteq(char, 1)
  mytester:assert(short >= 2)
  mytester:assert(int >= 2)
  mytester:assert(int >= short)
  mytester:assert(long >= 4)
  mytester:assert(long >= int)
  mytester:assert(double >= float)
  mytester:assert(half <= float)
end

function torchtest.isTensor()
   local t = torch.randn(3,4):half()
   print("\n Tensor:half() result: ", t)

   mytester:assert(torch.isTensor(t), 'error in isTensor')
   mytester:assert(torch.isTensor(t[1]), 'error in isTensor for subTensor')
   mytester:assert(not torch.isTensor(t[1][2]), 'false positive in isTensor')
   mytester:assert(torch.Tensor.isTensor(t), 'alias not working')
end
function torchtest.isStorage()
  local t = torch.randn(3,4)
  mytester:assert(torch.isStorage(t:storage()), 'error in isStorage')
  mytester:assert(not torch.isStorage(t), 'false positive in isStorage')
end

function torchtest.expand()
   local result = torch.Tensor()
   local tensor = torch.rand(8,1)
   local template = torch.rand(8,5)
   local target = template:size():totable()
   mytester:assertTableEq(tensor:expandAs(template):size():totable(), target, 'Error in expandAs')
   mytester:assertTableEq(tensor:expand(8,5):size():totable(), target, 'Error in expand')
   mytester:assertTableEq(tensor:expand(torch.LongStorage{8,5}):size():totable(), target, 'Error in expand using LongStorage')
   result:expandAs(tensor,template)
   mytester:assertTableEq(result:size():totable(), target, 'Error in expandAs using result')
   result:expand(tensor,8,5)
   mytester:assertTableEq(result:size():totable(), target, 'Error in expand using result')
   result:expand(tensor,torch.LongStorage{8,5})
   mytester:assertTableEq(result:size():totable(), target, 'Error in expand using result and LongStorage')
   mytester:asserteq((result:mean(2):view(8,1)-tensor):abs():max(), 0, 'Error in expand (not equal)')
end

function torchtest.repeatTensor()
   local result = torch.Tensor()
   local tensor = torch.rand(8,4)
   local size = {3,1,1}
   local sizeStorage = torch.LongStorage(size)
   local target = {3,8,4}
   mytester:assertTableEq(tensor:repeatTensor(unpack(size)):size():totable(), target, 'Error in repeatTensor')
   mytester:assertTableEq(tensor:repeatTensor(sizeStorage):size():totable(), target, 'Error in repeatTensor using LongStorage')
   result:repeatTensor(tensor,unpack(size))
   mytester:assertTableEq(result:size():totable(), target, 'Error in repeatTensor using result')
   result:repeatTensor(tensor,sizeStorage)
   mytester:assertTableEq(result:size():totable(), target, 'Error in repeatTensor using result and LongStorage')
   mytester:asserteq((result:mean(1):view(8,4)-tensor):abs():max(), 0, 'Error in repeatTensor (not equal)')
end

function torchtest.isSameSizeAs()
   local t1 = torch.Tensor(3, 4, 9, 10)
   local t2 = torch.Tensor(3, 4)
   local t3 = torch.Tensor(1, 9, 3, 3)
   local t4 = torch.Tensor(3, 4, 9, 10)

   mytester:assert(t1:isSameSizeAs(t2) == false, "wrong answer ")
   mytester:assert(t1:isSameSizeAs(t3) == false, "wrong answer ")
   mytester:assert(t1:isSameSizeAs(t4) == true, "wrong answer ")
end

   torch.setheaptracking(true)
   math.randomseed(os.time())
   precision = 1e-4
   mytester = torch.Tester()
   mytester:add(torchtest)
   mytester:run(tests)
