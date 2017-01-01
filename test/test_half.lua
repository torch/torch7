local mytester
local torchtest = torch.TestSuite()

-- Lua 5.2 compatibility
local loadstring = loadstring or load
local unpack = unpack or table.unpack

function torchtest.easy()
   local x=torch.randn(5, 6):half()
   mytester:assert(x:isContiguous(), 'x should be contiguous')
   mytester:assert(x:dim() == 2, 'x should have dim of 2')
   mytester:assert(x:nDimension() == 2, 'x should have nDimension of 2')
   mytester:assert(x:nElement() == 5 * 6, 'x should have 30 elements')
   local stride = x:stride()
   local expectedStride = torch.LongStorage{6,1}
   for i=1,stride:size() do
      mytester:assert(stride[i] == expectedStride[i], "stride is wrong")
   end

   x=x:t()
   mytester:assert(not x:isContiguous(), 'x transpose should not be contiguous')
   x=x:transpose(1,2)
   mytester:assert(x:isContiguous(), 'x should be contiguous after 2 transposes')

   local y=torch.HalfTensor()
   y:resizeAs(x:t()):copy(x:t())
   mytester:assert(x:isContiguous(), 'after resize and copy, x should be contiguous')
   mytester:assertTensorEq(y, x:t(), 0.001, 'copy broken after resizeAs')
   local z=torch.HalfTensor()
   z:resize(6, 5):copy(x:t())
   mytester:assertTensorEq(y, x:t(), 0.001, 'copy broken after resize')
end

function torchtest.narrowSub()
   local x = torch.randn(5, 6):half()
   local narrow = x:narrow(1, 2, 3)
   local sub = x:sub(2, 4)
   mytester:assertTensorEq(narrow, sub, 0.001, 'narrow not equal to sub')
end

function torchtest.selectClone()
   local x = torch.zeros(5, 6)
   x:select(1,2):fill(2)
   x=x:half()
   local y=x:clone()
   mytester:assertTensorEq(x, y, 0.001, 'not equal after select and clone')
   x:select(1,1):fill(3)
   mytester:assert(y[1][1] == 0, 'clone broken')
end

torch.setheaptracking(true)
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(torchtest)
mytester:run(tests)
