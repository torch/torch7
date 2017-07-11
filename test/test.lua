--require 'torch'

local mytester
local torchtest = torch.TestSuite()
local msize = 100
local precision

-- Lua 5.2 compatibility
local loadstring = loadstring or load
local unpack = unpack or table.unpack

local function maxdiff(x,y)
   local d = x-y
   if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.FloatTensor' then
      return d:abs():max()
   else
      local dd = torch.Tensor():resize(d:size()):copy(d)
      return dd:abs():max()
   end
end

-- workarounds for non-existant functions
function torch.HalfTensor:__sub(other)
   return (self:real() - other:real()):half()
end

function torch.HalfTensor:mean(dim)
   return self:real():mean(dim):half()
end

function torch.HalfTensor:abs()
   return self:real():abs():half()
end

function torch.HalfTensor:max()
   return self:real():max()
end

function torch.HalfTensor:add(a, b)
   return (self:real():add(a, b:real())):half()
end

function torch.HalfTensor:reshape(a, b)
   return (self:real():reshape(a, b)):half()
end

function torch.HalfTensor:fill(a)
   return self:real():fill(a):half()
end

function torchtest.dot()
   local types = {
      ['torch.DoubleTensor'] = 1e-8, -- for ddot
      ['torch.FloatTensor']  = 1e-4, -- for sdot
   }
   for tname, prec in pairs(types) do
      local v1 = torch.randn(100):type(tname)
      local v2 = torch.randn(100):type(tname)

      local res1 = torch.dot(v1,v2)

      local res2 = 0
      for i = 1,v1:size(1) do
         res2 = res2 + v1[i] * v2[i]
      end

      local err = math.abs(res1-res2)

      mytester:assertlt(err, prec, 'error in torch.dot (' .. tname .. ')')
   end
end

local genericSingleOpTest = [[
   -- [res] torch.functionname([res,] x)
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.functionname(m1[{ 4,{} }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.functionname(m1[4][i])
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerrc = 0
   for i = 1, err:size(1) do
      if err[i] > maxerrc then
         maxerrc = err[i]
      end
   end

   -- non-contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.functionname(m1[{ {}, 4 }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.functionname(m1[i][4])
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerrnc = 0
   for i = 1, err:size(1) do
      if err[i] > maxerrnc then
         maxerrnc = err[i]
      end
   end
   return maxerrc, maxerrnc
--]]

function torchtest.sin()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'sin'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.sinh()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'sinh'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.asin()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'asin'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.cos()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'cos'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.cosh()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'cosh'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.acos()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'acos'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.tan()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'tan'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.tanh()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'tanh'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.atan()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'atan'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.log()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'log'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.sqrt()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'sqrt'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.rsqrt()
   local function TH_rsqrt(x)
      return 1 / math.sqrt(x)
   end

   local f
   local t = genericSingleOpTest:gsub('functionname', 'rsqrt'):gsub('math.rsqrt', 'TH_rsqrt')
   local env = { TH_rsqrt=TH_rsqrt, torch=torch, math=math }
   if not setfenv then -- Lua 5.2
      f = load(t, 'test', 't', env)
   else
      f = loadstring(t)
      setfenv(f, env)
   end

   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.sigmoid()
   -- can't use genericSingleOpTest, since `math.sigmoid` doesn't exist, have to use
   -- `torch.sigmoid` instead
   local inputValues = {-1000,-1,0,0.5,1,2,1000}
   local expectedOutput = {0.0000, 0.2689, 0.5, 0.6225, 0.7311, 0.8808, 1.000}

   local precision_4dps = 0.0002

   -- float
   local inputFT = torch.FloatTensor(inputValues)
   local expectedFT = torch.FloatTensor(expectedOutput)
   mytester:assertlt((torch.sigmoid(inputFT) - expectedFT):abs():max(), precision_4dps, 'error in torch.sigmoid - single')
   mytester:assertlt((inputFT - torch.FloatTensor(inputValues)):abs():max(), precision_4dps, 'error in torch.sigmoid - single')
   local sigmoidFT = torch.FloatTensor(inputValues):sigmoid()
   mytester:assertlt((sigmoidFT - expectedFT):abs():max(), precision_4dps, 'error in torch.sigmoid - single')

   -- double
   local inputDT = torch.DoubleTensor(inputValues)
   local expectedDT = torch.DoubleTensor(expectedOutput)
   mytester:assertlt((torch.sigmoid(inputDT) - expectedDT):abs():max(), precision_4dps, 'error in torch.sigmoid - double')
   mytester:assertlt((inputDT - torch.DoubleTensor(inputValues)):abs():max(), precision_4dps, 'error in torch.sigmoid - double')
   local sigmoidDT = torch.DoubleTensor(inputValues):sigmoid()
   mytester:assertlt((sigmoidDT - expectedDT):abs():max(), precision_4dps, 'error in torch.sigmoid - double')
end

function torchtest.exp()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'exp'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.floor()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'floor'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.ceil()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'ceil'))
   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.frac()
   local function TH_frac(x)
      return math.fmod(x, 1)
   end

   local f
   local t = genericSingleOpTest:gsub('functionname', 'frac'):gsub('math.frac', 'TH_frac')
   local env = { TH_frac=TH_frac, torch=torch, math=math }
   if not setfenv then -- Lua 5.2
      f = load(t, 'test', 't', env)
   else
      f = loadstring(t)
      setfenv(f, env)
   end

   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.trunc()
   local function TH_trunc(x)
      return x - math.fmod(x, 1)
   end

   local f
   local t = genericSingleOpTest:gsub('functionname', 'trunc'):gsub('math.trunc', 'TH_trunc')
   local env = { TH_trunc=TH_trunc, torch=torch, math=math }
   if not setfenv then -- Lua 5.2
      f = load(t, 'test', 't', env)
   else
      f = loadstring(t)
      setfenv(f, env)
   end

   local maxerrc, maxerrnc = f()
   mytester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   mytester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function torchtest.round()
   -- [res] torch.round([res,] x)
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.round(m1[{ 4,{} }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.floor(m1[4][i]+0.5)
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerrc = 0
   for i = 1, err:size(1) do
      if err[i] > maxerrc then
         maxerrc = err[i]
      end
   end
   mytester:assertlt(maxerrc, precision, 'error in torch.round - contiguous')

   -- non-contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.round(m1[{ {}, 4 }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.floor(m1[i][4]+0.5)
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerrnc = 0
   for i = 1, err:size(1) do
      if err[i] > maxerrnc then
         maxerrnc = err[i]
      end
   end
   mytester:assertlt(maxerrnc, precision, 'error in torch.round - non-contiguous')
end

function torchtest.max()  -- torch.max([resval, resind,] x [,dim])

   -- TH_TENSOR_BASE
   local m1 = torch.Tensor(8,2):fill(3):select(2, 1)
   local resval, resind = torch.max(m1, 1)
   mytester:assert(resind[1] == 1)

   -- torch.max( x )
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.max(m1)
   local res2 = m1[1][1]
   for i = 1,m1:size(1) do
      for j = 1,m1:size(2) do
         if m1[i][j] > res2 then
            res2 = m1[i][j]
         end
      end
   end
   local err = res1 - res2
   mytester:assertlt(err, precision, 'error in torch.max - contiguous')

   -- non-contiguous
   local m1 = torch.randn(10,10,10)
   local m2 = m1[{{}, 4, {}}]
   local res1 = torch.max(m2)
   local res2 = m2[1][1]
   for i = 1,m2:size(1) do
      for j = 1,m2:size(2) do
         if m2[i][j] > res2 then
            res2 = m2[i][j]
         end
      end
   end
   local err = res1 - res2
   mytester:assertlt(err, precision, 'error in torch.max - non-contiguous')

   -- torch.max([resval, resind,] x ,dim])
   function lua_max(t, dim)
      assert(t:nDimension() == 2)
      max_val = t:narrow(dim, 1, 1):clone()
      max_ind = t:narrow(dim, 1, 1):clone():long():fill(1)
      other = 3 - dim
      for i = 1, t:size(other) do
         for j = 1, t:size(dim) do
            val = t:select(other, i):select(dim, j)
            max = max_val:select(other, i):select(dim, 1)
            if val > max then
               max_val:select(other, i):fill(val)
               max_ind:select(other, i):fill(j)
            end
         end
      end
      return max_val, max_ind
   end

   local m1 = torch.randn(100,100)
   for dim = 1,2 do
      local res1val, res1ind = torch.max(m1, dim)
      local res2val, res2ind = lua_max(m1, dim)
      mytester:asserteq((res1val-res2val):abs():max(), 0, 'error in torch.max')
      mytester:asserteq((res1ind-res2ind):abs():max(), 0, 'error in torch.max')
   end

   -- NaNs
   for index in pairs{1, 5, 100} do
      local m1 = torch.randn(100)
      m1[index] = 0/0
      local res1val, res1ind = torch.max(m1, 1)
      mytester:assert(res1val[1] ~= res1val[1], 'error in torch.max (value) - NaNs')
      mytester:assert(res1ind[1] == index, 'error in torch.max (index) - NaNs')
      local res1val = torch.max(m1)
      mytester:assert(res1val ~= res1val, 'error in torch.max - NaNs')
   end

   -- dim == nDim -1
   local a = torch.Tensor({{1,2},{3,4}}):select(2, 1)
   local aval, aind = torch.max(a, 1)
   mytester:assert(aval[1] == 3)
   mytester:assert(aind[1] == 2)

   local b = torch.Tensor({{{1,2},{3,4}},{{5,6},{7,8}}}):select(3, 1)
   local bval, bind = torch.max(b, 2)
   mytester:assert(bval[1][1] == 3)
   mytester:assert(bind[1][1] == 2)
   mytester:assert(bval[2][1] == 7)
   mytester:assert(bind[2][1] == 2)
end

function torchtest.min()  -- torch.min([resval, resind,] x [,dim])
   -- torch.min( x )
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.min(m1)
   local res2 = m1[1][1]
   for i = 1,m1:size(1) do
      for j = 1,m1:size(2) do
         if m1[i][j] < res2 then
            res2 = m1[i][j]
         end
      end
   end
   local err = res1 - res2
   mytester:assertlt(err, precision, 'error in torch.min - contiguous')
   -- non-contiguous
   local m1 = torch.randn(10,10,10)
   local m2 = m1[{{}, 4, {}}]
   local res1 = torch.min(m2)
   local res2 = m2[1][1]
   for i = 1,m2:size(1) do
      for j = 1,m2:size(2) do
         if m2[i][j] < res2 then
            res2 = m2[i][j]
         end
      end
   end
   local err = res1 - res2
   mytester:assertlt(err, precision, 'error in torch.min - non-contiguous')

   -- torch.max([resval, resind,] x ,dim])
   function lua_min(t, dim)
      assert(t:nDimension() == 2)
      max_val = t:narrow(dim, 1, 1):clone()
      max_ind = t:narrow(dim, 1, 1):clone():long():fill(1)
      other = 3 - dim
      for i = 1, t:size(other) do
         for j = 1, t:size(dim) do
            val = t:select(other, i):select(dim, j)
            max = max_val:select(other, i):select(dim, 1)
            if val < max then
               max_val:select(other, i):fill(val)
               max_ind:select(other, i):fill(j)
            end
         end
      end
      return max_val, max_ind
   end

   local m1 = torch.randn(100,100)
   for dim = 1,2 do
      local res1val, res1ind = torch.min(m1, dim)
      local res2val, res2ind = lua_min(m1, dim)
      mytester:asserteq((res1val-res2val):abs():max(), 0, 'error in torch.max')
      mytester:asserteq((res1ind-res2ind):abs():max(), 0, 'error in torch.max')
   end

   -- NaNs
   for index in pairs{1, 5, 100} do
      local m1 = torch.randn(100)
      m1[index] = 0/0
      local res1val, res1ind = torch.min(m1, 1)
      mytester:assert(res1val[1] ~= res1val[1], 'error in torch.min (value) - NaNs')
      mytester:assert(res1ind[1] == index, 'error in torch.min (index) - NaNs')
      local res1val = torch.min(m1)
      mytester:assert(res1val ~= res1val, 'error in torch.min - NaNs')
   end

   -- TH_TENSOR_BASE
   local m1 = torch.Tensor(4):fill(3)
   local resval, resind = torch.min(m1, 1)
   mytester:assert(resind[1] == 1)
end

function torchtest.cmax()
  -- Two tensors.
  local a = torch.rand(msize, msize)
  local b = torch.rand(msize, msize)
  local c = torch.cmax(a, b)
  local expected_c = torch.zeros(msize, msize)
  expected_c:map2(a, b, function(_, a, b) return math.max(a, b) end)
  mytester:assertTensorEq(expected_c, c, 0,
                          'error in torch.cmax(tensor, tensor)')

  -- Tensor and scalar.
  local v = torch.uniform()
  c = torch.cmax(a, v)
  expected_c:map(a, function(_, a) return math.max(a, v) end)
  mytester:assertTensorEq(expected_c, c, 0,
                          'error in torch.cmax(tensor, scalar).')
end

function torchtest.cmin()
  -- Two tensors.
  local a = torch.rand(msize, msize)
  local b = torch.rand(msize, msize)
  local c = torch.cmin(a, b)
  local expected_c = torch.zeros(msize, msize)
  expected_c:map2(a, b, function(_, a, b) return math.min(a, b) end)
  mytester:assertTensorEq(expected_c, c, 0,
                          'error in torch.cmin(tensor, tensor)')

  -- Tensor and scalar.
  local v = torch.uniform()
  c = torch.cmin(a, v)
  expected_c:map(a, function(_, a) return math.min(a, v) end)
  mytester:assertTensorEq(expected_c, c, 0,
                          'error in torch.cmin(tensor, scalar).')
end

function torchtest.lerp()
   local function TH_lerp(a, b, weight)
      return a + weight * (b-a);
   end

   local a = torch.rand(msize, msize)
   local b = torch.rand(msize, msize)
   local w = math.random()
   local result = torch.lerp(a, b, w)
   local expected = a:new()
   expected:map2(a, b, function(_, a, b) return TH_lerp(a, b, w) end)
   mytester:assertTensorEq(expected, result, precision, 'error in torch.lerp(tensor, tensor, weight)')

   local a = (math.random()*2-1) * 100000
   local b = (math.random()*2-1) * 100000
   local w = math.random()
   local result = torch.lerp(a, b, w)
   local expected = TH_lerp(a, b, w)
   mytester:assertalmosteq(expected, result, precision, 'error in torch.lerp(scalar, scalar, weight)')
end

for i, v in ipairs{{10}, {5, 5}} do
   torchtest['allAndAny' .. i] =
      function ()
           local x = torch.ones(unpack(v)):byte()
           mytester:assert(x:all(), 'error in all()')
           mytester:assert(x:any(), 'error in any()')

           x[3] = 0
           mytester:assert(not x:all(), 'error in all()')
           mytester:assert(x:any(), 'error in any()')

           x:zero()
           mytester:assert(not x:all(), 'error in all()')
           mytester:assert(not x:any(), 'error in any()')

           x:fill(2)
           mytester:assert(x:all(), 'error in all()')
           mytester:assert(x:any(), 'error in any()')
       end
end

function torchtest.mv()
   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.mv(m1,v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(1) do
      for j = 1,m1:size(2) do
         res2[i] = res2[i] + m1[i][j] * v1[j]
      end
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.mv')
end

function torchtest.fill()
   local types = {
      'torch.ByteTensor',
      'torch.CharTensor',
      'torch.ShortTensor',
      'torch.IntTensor',
      'torch.FloatTensor',
      'torch.DoubleTensor',
      'torch.LongTensor',
   }

   for k,t in ipairs(types) do
      -- [res] torch.fill([res,] tensor, value)
      local m1 = torch.ones(100,100):type(t)
      local res1 = m1:clone()
      res1[{ 3,{} }]:fill(2)

      local res2 = m1:clone()
      for i = 1,m1:size(1) do
	 res2[{ 3,i }] = 2
      end

      local err = (res1-res2):double():abs():max()

      mytester:assertlt(err, precision, 'error in torch.fill - contiguous')

      local m1 = torch.ones(100,100):type(t)
      local res1 = m1:clone()
      res1[{ {},3 }]:fill(2)

      local res2 = m1:clone()
      for i = 1,m1:size(1) do
	 res2[{ i,3 }] = 2
      end

      local err = (res1-res2):double():abs():max()

      mytester:assertlt(err, precision, 'error in torch.fill - non contiguous')
   end
end

function torchtest.add()
   local types = {
      'torch.ByteTensor',
      'torch.CharTensor',
      'torch.ShortTensor',
      'torch.IntTensor',
      'torch.FloatTensor',
      'torch.DoubleTensor',
      'torch.LongTensor',
   }

   for k,t in ipairs(types) do
       -- [res] torch.add([res,] tensor1, tensor2)
       local m1 = torch.randn(100,100):type(t)
       local v1 = torch.randn(100):type(t)

       local res1 = torch.add(m1[{ 4,{} }],v1)

       local res2 = res1:clone():zero()
       for i = 1,m1:size(2) do
           res2[i] = m1[4][i] + v1[i]
       end

       local err = (res1-res2):double():abs():max()

       mytester:assertlt(err, precision, 'error in torch.add - contiguous' .. ' ' .. t)

       local m1 = torch.randn(100,100):type(t)
       local v1 = torch.randn(100):type(t)

       local res1 = torch.add(m1[{ {},4 }],v1)

       local res2 = res1:clone():zero()
       for i = 1,m1:size(1) do
           res2[i] = m1[i][4] + v1[i]
       end

       local err = (res1-res2):double():abs():max()

       mytester:assertlt(err, precision, 'error in torch.add - non contiguous' .. ' ' .. t)

       -- [res] torch.add([res,] tensor, value)
       local m1 = torch.randn(10,10):type(t)
       local res1 = m1:clone()
       res1[{ 3,{} }]:add(2)

       local res2 = m1:clone()
       for i = 1,m1:size(1) do
           res2[{ 3,i }] = res2[{ 3,i }] + 2
       end

       local err = (res1-res2):double():abs():max()

       mytester:assertlt(err, precision, 'error in torch.add - scalar, contiguous' .. ' ' .. t)

       local m1 = torch.randn(10,10)
       local res1 = m1:clone()
       res1[{ {},3 }]:add(2)

       local res2 = m1:clone()
       for i = 1,m1:size(1) do
           res2[{ i,3 }] = res2[{ i,3 }] + 2
       end

       local err = (res1-res2):abs():max()

       mytester:assertlt(err, precision, 'error in torch.add - scalar, non contiguous' .. ' ' .. t)

       -- [res] torch.add([res,] tensor1, value, tensor2)
   end
end

function torchtest.csub()
   local rngState = torch.getRNGState()
   torch.manualSeed(123)

   local a = torch.randn(100,90)
   local b = a:clone():normal()

   local res_add = torch.add(a, -1, b)
   local res_csub = a:clone()
   res_csub:csub(b)

   mytester:assertlt((res_add - res_csub):abs():max(), 0.00001)

   local _ = torch.setRNGState(rngState)
end

function torchtest.csub_scalar()
   local rngState = torch.getRNGState()
   torch.manualSeed(123)

   local a = torch.randn(100,100)

   local scalar = 123.5
   local res_add = torch.add(a, -scalar)
   local res_csub = a:clone()
   res_csub:csub(scalar)

   mytester:assertlt((res_add - res_csub):abs():max(), 0.00001)

   local _ = torch.setRNGState(rngState)
end

function torchtest.neg()
   local rngState = torch.getRNGState()
   torch.manualSeed(123)

   local a = torch.randn(100,90)
   local zeros = torch.Tensor():resizeAs(a):zero()

   local res_add = torch.add(zeros, -1, a)
   local res_neg = a:clone()
   res_neg:neg()

   mytester:assertlt((res_add - res_neg):abs():max(), 0.00001)

   local _ = torch.setRNGState(rngState)
end

function torchtest.cinv()
   local rngState = torch.getRNGState()
   torch.manualSeed(123)

   local a = torch.randn(100,89)
   local zeros = torch.Tensor():resizeAs(a):zero()

   local res_pow = torch.pow(a, -1)
   local res_inv = a:clone()
   res_inv:cinv()

   mytester:assertlt((res_pow - res_inv):abs():max(), 0.00001)

   local _ = torch.setRNGState(rngState)
end

function torchtest.mul()
   local types = {
      'torch.ByteTensor',
      'torch.CharTensor',
      'torch.ShortTensor',
      'torch.IntTensor',
      'torch.FloatTensor',
      'torch.DoubleTensor',
      'torch.LongTensor',
   }

   for k,t in ipairs(types) do
       local m1 = torch.randn(10,10):type(t)
       local res1 = m1:clone()

       res1[{ {},3 }]:mul(2)

       local res2 = m1:clone()
       for i = 1,m1:size(1) do
           res2[{ i,3 }] = res2[{ i,3 }] * 2
       end

       local err = (res1-res2):double():abs():max()

       mytester:assertlt(err, precision, 'error in torch.mul - scalar, non contiguous' .. ' ' .. t)
   end
end

function torchtest.div()
    local types = {
        'torch.ByteTensor',
        'torch.CharTensor',
        'torch.ShortTensor',
        'torch.IntTensor',
        'torch.FloatTensor',
        'torch.DoubleTensor',
        'torch.LongTensor',
    }

    for k,t in ipairs(types) do

        local m1 = torch.Tensor(10,10):uniform(0,10):type(t)
        local res1 = m1:clone()

        res1[{ {},3 }]:div(2)

        local res2 = m1:clone()
        for i = 1,m1:size(1) do
            local ok = pcall(function() res2[{ i,3 }] = res2[{ i,3 }] / 2 end)
            if not ok then
               res2[{ i,3 }] = torch.floor(res2[{ i,3 }] / 2)
            end
        end

        local err = (res1-res2):double():abs():max()

        mytester:assertlt(err, precision, 'error in torch.div - scalar, non contiguous' .. ' ' .. t)
    end
end

function torchtest.lshift()
   local m1 = torch.LongTensor(10,10):random(0,100)
   local res1 = m1:clone()

   local q = 2
   local f = math.pow(2, q)
   res1[{ {},3 }]:lshift(q)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] * f
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.lshift - scalar, non contiguous')

   local m1 = torch.LongTensor(10,10):random(0,100)
   local res1 = m1:clone()

   local q = 2
   res1:lshift(q)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      for j = 1,m1:size(1) do
         res2[{ i,j }] = res2[{ i,j }] * f
      end
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.lshift - scalar, contiguous')
end

function torchtest.rshift()
   local m1 = torch.LongTensor(10,10):random(0,100)
   local res1 = m1:clone()

   local q = 2
   local f = math.pow(2, q)
   res1[{ {},3 }]:rshift(q)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = math.floor(res2[{ i,3 }] / f)
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.rshift - scalar, non contiguous')

   local m1 = torch.LongTensor(10,10):random(0,100)
   local res1 = m1:clone()

   local q = 2
   res1:rshift(q)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      for j = 1,m1:size(1) do
         res2[{ i,j }] = math.floor(res2[{ i,j }] / f)
      end
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.rshift - scalar, contiguous')
end

function torchtest.fmod()
   local m1 = torch.Tensor(10,10):uniform(-10, 10)
   local res1 = m1:clone()

   local q = 2.1
   res1[{ {},3 }]:fmod(q)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = math.fmod(res2[{ i,3 }], q)
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.fmod - scalar, non contiguous')
end

function torchtest.remainder()
   local m1 = torch.Tensor(10, 10):uniform(-10, 10)
   local res1 = m1:clone()

   local q = 2.1
   res1[{ {},3 }]:remainder(q)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] % q
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.remainder - scalar, non contiguous')
end

function torchtest.bitand()
   local m1 = torch.LongTensor(10,10):random(0,100)
   local res1 = m1:clone()

   local val = 32 -- This should be a power of 2
   res1[{ {},3 }]:bitand(val - 1)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] % val
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.bitand - scalar, non contiguous')

   local m1 = torch.LongTensor(10,10):random(0,100)
   local res1 = m1:clone()

   res1:bitand(val - 1)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      for j = 1,m1:size(1) do
         res2[{ i,j }] = res2[{ i,j }] % val
      end
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.bitand - scalar, contiguous')
end

function torchtest.bitor()
   local m1 = torch.LongTensor(10,10):random(0,10000)
   local res1 = m1:clone()

   local val = 32 -- This should be a power of 2
   res1[{ {},3 }]:bitor(val-1)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = math.floor(res2[{ i,3 }] / val) * val + (val - 1)
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.bitor - scalar, non contiguous')

   local m1 = torch.LongTensor(10,10):random(0,10000)
   local res1 = m1:clone()

   res1:bitor(val - 1)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      for j = 1,m1:size(1) do
         res2[{ i,j }] = math.floor(res2[{ i,j }] / val) * val + (val - 1)
      end
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.bitor - scalar, contiguous')
end

function torchtest.cbitxor()
   local t1 = torch.LongTensor(10,10):random(0,10000)
   local t2 = torch.LongTensor(10,10):random(10001,20000)

   -- Perform xor swap and check results
   local t3 = torch.cbitxor(t1, t2)
   local r1 = torch.cbitxor(t3, t2)
   local r2 = torch.cbitxor(t3, t1)

   local err1 = (r1 - t1):abs():max()
   local err2 = (r2 - t2):abs():max()
   mytester:assertlt(err1 + err2, precision, 'error in torch.cbitxor contiguous')
end

function torchtest.mm()
   -- helper function
   local function matrixmultiply(mat1,mat2)
      local n = mat1:size(1)
      local m = mat1:size(2)
      local p = mat2:size(2)
      local res = torch.zeros(n,p)
      for i = 1, n do
         for j = 1, p do
            local sum = 0
            for k = 1, m do
               sum = sum + mat1[i][k]*mat2[k][j]
            end
            res[i][j] = sum
         end
      end
      return res
   end

   -- contiguous case
   local n, m, p = 10, 10, 5
   local mat1 = torch.randn(n,m)
   local mat2 = torch.randn(m,p)
   local res = torch.mm(mat1,mat2)

   local res2 = matrixmultiply(mat1,mat2)
   mytester:assertTensorEq(res,res2,precision,'error in torch.mm')

   -- non contiguous case 1
   local n, m, p = 10, 10, 5
   local mat1 = torch.randn(n,m)
   local mat2 = torch.randn(p,m):t()
   local res = torch.mm(mat1,mat2)

   local res2 = matrixmultiply(mat1,mat2)
   mytester:assertTensorEq(res,res2,precision,'error in torch.mm, non contiguous')

   -- non contiguous case 2
   local n, m, p = 10, 10, 5
   local mat1 = torch.randn(m,n):t()
   local mat2 = torch.randn(m,p)
   local res = torch.mm(mat1,mat2)

   local res2 = matrixmultiply(mat1,mat2)
   mytester:assertTensorEq(res,res2,precision,'error in torch.mm, non contiguous')

   -- non contiguous case 3
   local n, m, p = 10, 10, 5
   local mat1 = torch.randn(m,n):t()
   local mat2 = torch.randn(p,m):t()
   local res = torch.mm(mat1,mat2)

   local res2 = matrixmultiply(mat1,mat2)
   mytester:assertTensorEq(res,res2,precision,'error in torch.mm, non contiguous')

   -- test with zero stride
   local n, m, p = 10, 10, 5
   local mat1 = torch.randn(n,m)
   local mat2 = torch.randn(m,1):expand(m,p)
   local res = torch.mm(mat1,mat2)

   local res2 = matrixmultiply(mat1,mat2)
   mytester:assertTensorEq(res,res2,precision,'error in torch.mm, non contiguous, zero stride')

end

function torchtest.bmm()
   local num_batches = 10
   local M, N, O = 23, 8, 12
   local b1 = torch.randn(num_batches, M, N)
   local b2 = torch.randn(num_batches, N, O)
   local res = torch.bmm(b1, b2)

   for i = 1, num_batches do
     local r = torch.mm(b1[i], b2[i])
     mytester:assertTensorEq(r, res[i], precision, 'result matrix ' .. i .. ' wrong')
   end
end

function torchtest.addbmm()
   local num_batches = 10
   local M, N, O = 12, 8, 5
   local b1 = torch.randn(num_batches, M, N)
   local b2 = torch.randn(num_batches, N, O)
   local res = torch.bmm(b1, b2)
   local res2 = torch.Tensor():resizeAs(res[1]):zero()

   res2:addbmm(b1,b2)
   mytester:assertTensorEq(res2, res:sum(1)[1], precision, 'addbmm result wrong')

   res2:addbmm(1,b1,b2)
   mytester:assertTensorEq(res2, res:sum(1)[1]*2, precision, 'addbmm result wrong')

   res2:addbmm(1,res2,.5,b1,b2)
   mytester:assertTensorEq(res2, res:sum(1)[1]*2.5, precision, 'addbmm result wrong')

   local res3 = torch.addbmm(1,res2,0,b1,b2)
   mytester:assertTensorEq(res3, res2, precision, 'addbmm result wrong')

   local res4 = torch.addbmm(1,res2,.5,b1,b2)
   mytester:assertTensorEq(res4, res:sum(1)[1]*3, precision, 'addbmm result wrong')

   local res5 = torch.addbmm(0,res2,1,b1,b2)
   mytester:assertTensorEq(res5, res:sum(1)[1], precision, 'addbmm result wrong')

   local res6 = torch.addbmm(.1,res2,.5,b1,b2)
   mytester:assertTensorEq(res6, res2*.1 + res:sum(1)*.5, precision, 'addbmm result wrong')
end

function torchtest.baddbmm()
   local num_batches = 10
   local M, N, O = 12, 8, 5
   local b1 = torch.randn(num_batches, M, N)
   local b2 = torch.randn(num_batches, N, O)
   local res = torch.bmm(b1, b2)
   local res2 = torch.Tensor():resizeAs(res):zero()

   res2:baddbmm(b1,b2)
   mytester:assertTensorEq(res2, res, precision, 'baddbmm result wrong')

   res2:baddbmm(1,b1,b2)
   mytester:assertTensorEq(res2, res*2, precision, 'baddbmm result wrong')

   res2:baddbmm(1,res2,.5,b1,b2)
   mytester:assertTensorEq(res2, res*2.5, precision, 'baddbmm result wrong')

   local res3 = torch.baddbmm(1,res2,0,b1,b2)
   mytester:assertTensorEq(res3, res2, precision, 'baddbmm result wrong')

   local res4 = torch.baddbmm(1,res2,.5,b1,b2)
   mytester:assertTensorEq(res4, res*3, precision, 'baddbmm result wrong')

   local res5 = torch.baddbmm(0,res2,1,b1,b2)
   mytester:assertTensorEq(res5, res, precision, 'baddbmm result wrong')

   local res6 = torch.baddbmm(.1,res2,.5,b1,b2)
   mytester:assertTensorEq(res6, res2*.1 + res*.5, precision, 'baddbmm result wrong')
end

function torchtest.clamp()
   local m1 = torch.rand(100):mul(5):add(-2.5)  -- uniform in [-2.5, 2.5]
   -- just in case we're extremely lucky:
   local min_val = -1
   local max_val = 1
   m1[1] = min_val
   m1[2] = max_val
   local res1 = m1:clone()

   res1:clamp(min_val, max_val)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      if res2[i] > max_val then
         res2[i] = max_val
      elseif res2[i] < min_val then
         res2[i] = min_val
      end
   end

   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.clamp - scalar, non contiguous')
end

function torchtest.pow() -- [res] torch.pow([res,] x)
   -- base - tensor, exponent - number
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.pow(m1[{ 4,{} }], 3)
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.pow(m1[4][i], 3)
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      if err[i] > maxerr then
         maxerr = err[i]
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.pow - contiguous')

   -- non-contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.pow(m1[{ {}, 4 }], 3)
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.pow(m1[i][4], 3)
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      if err[i] > maxerr then
         maxerr = err[i]
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.pow - non-contiguous')

   -- base - number, exponent - tensor
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.pow(3, m1[{ 4,{} }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.pow(3, m1[4][i])
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      if err[i] > maxerr then
         maxerr = err[i]
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.pow - contiguous')

   -- non-contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.pow(3, m1[{ {}, 4 }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.pow(3, m1[i][4])
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      if err[i] > maxerr then
         maxerr = err[i]
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.pow - non-contiguous')
end

function torchtest.cdiv()
    local types = {
        'torch.ByteTensor',
        'torch.CharTensor',
        'torch.ShortTensor',
        'torch.IntTensor',
        'torch.FloatTensor',
        'torch.DoubleTensor',
        'torch.LongTensor',
    }

    for k,t in ipairs(types) do

        -- [res] torch.cdiv([res,] tensor1, tensor2)
        -- contiguous
        local m1 = torch.Tensor(10, 10, 10):uniform(0,10):type(t)
        local m2 = torch.Tensor(10, 10 * 10):uniform(0,10):type(t)
        m2[m2:eq(0)] = 2
        local sm1 = m1[{4, {}, {}}]
        local sm2 = m2[{4, {}}]
        local res1 = torch.cdiv(sm1, sm2)
        local res2 = res1:clone():zero()
        for i = 1,sm1:size(1) do
            for j = 1, sm1:size(2) do
                local idx1d = (((i-1)*sm1:size(1)))+j
                local ok = pcall(function() res2[i][j] = sm1[i][j] / sm2[idx1d] end)
                if not ok then
                   res2[i][j] = torch.floor(sm1[i][j] / sm2[idx1d])
                end
            end
        end
        local err = res1:clone():zero()
        -- find absolute error
        for i = 1, res1:size(1) do
            for j = 1, res1:size(2) do
                err[i][j] = math.abs(res1[i][j] - res2[i][j])
            end
        end
        -- find maximum element of error
        local maxerr = 0
        for i = 1, err:size(1) do
            for j = 1, err:size(2) do
                if err[i][j] > maxerr then
                    maxerr = err[i][j]
                end
            end
        end
        mytester:assertlt(maxerr, precision, 'error in torch.cdiv - contiguous' .. ' ' .. t)

        -- non-contiguous
        local m1 = torch.Tensor(10, 10, 10):uniform(0,10):type(t)
        local m2 = torch.Tensor(10 * 10, 10 * 10):uniform(0,10):type(t)
        m2[m2:eq(0)] = 2
        local sm1 = m1[{{}, 4, {}}]
        local sm2 = m2[{{}, 4}]
        local res1 = torch.cdiv(sm1, sm2)
        local res2 = res1:clone():zero()
        for i = 1,sm1:size(1) do
            for j = 1, sm1:size(2) do
                local idx1d = (((i-1)*sm1:size(1)))+j
                local ok = pcall(function() res2[i][j] = sm1[i][j] / sm2[idx1d] end)
                if not ok then
                   res2[i][j] = torch.floor(sm1[i][j] / sm2[idx1d])
                end
            end
        end
        local err = res1:clone():zero()
        -- find absolute error
        for i = 1, res1:size(1) do
            for j = 1, res1:size(2) do
                err[i][j] = math.abs(res1[i][j] - res2[i][j])
            end
        end
        -- find maximum element of error
        local maxerr = 0
        for i = 1, err:size(1) do
            for j = 1, err:size(2) do
                if err[i][j] > maxerr then
                    maxerr = err[i][j]
                end
            end
        end
        mytester:assertlt(maxerr, precision, 'error in torch.cdiv - non-contiguous' .. ' ' .. t)
   end
end

function torchtest.cfmod()
   -- contiguous
   local m1 = torch.Tensor(10, 10, 10):uniform(-10, 10)
   local m2 = torch.Tensor(10, 10 * 10):uniform(-3, 3)
   local sm1 = m1[{4, {}, {}}]
   local sm2 = m2[{4, {}}]
   local res1 = torch.cfmod(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
         local idx1d = (((i-1)*sm1:size(1)))+j
         res2[i][j] = math.fmod(sm1[i][j], sm2[idx1d])
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
         err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
         if err[i][j] > maxerr then
            maxerr = err[i][j]
         end
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.cfmod - contiguous')

   -- non-contiguous
   local m1 = torch.Tensor(10, 10, 10):uniform(-10, 10)
   local m2 = torch.Tensor(10 * 10, 10 * 10):uniform(-3, 3)
   local sm1 = m1[{{}, 4, {}}]
   local sm2 = m2[{{}, 4}]
   local res1 = torch.cfmod(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
         local idx1d = (((i-1)*sm1:size(1)))+j
         res2[i][j] = math.fmod(sm1[i][j], sm2[idx1d])
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
         err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
         if err[i][j] > maxerr then
            maxerr = err[i][j]
         end
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.cfmod - non-contiguous')
end

function torchtest.cremainder()
   -- contiguous
   local m1 = torch.Tensor(10, 10, 10):uniform(-10, 10)
   local m2 = torch.Tensor(10, 10 * 10):uniform(-3, 3)
   local sm1 = m1[{4, {}, {}}]
   local sm2 = m2[{4, {}}]
   local res1 = torch.cremainder(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
         local idx1d = (((i-1)*sm1:size(1)))+j
         res2[i][j] = sm1[i][j] % sm2[idx1d]
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
         err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
         if err[i][j] > maxerr then
            maxerr = err[i][j]
         end
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.cremainder - contiguous')

   -- non-contiguous
   local m1 = torch.Tensor(10, 10, 10):uniform(-10, 10)
   local m2 = torch.Tensor(10 * 10, 10 * 10):uniform(-3, 3)
   local sm1 = m1[{{}, 4, {}}]
   local sm2 = m2[{{}, 4}]
   local res1 = torch.cremainder(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
         local idx1d = (((i-1)*sm1:size(1)))+j
         res2[i][j] = sm1[i][j] % sm2[idx1d]
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
         err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
         if err[i][j] > maxerr then
            maxerr = err[i][j]
         end
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.cremainder - non-contiguous')
end

function torchtest.cmul()
    local types = {
        'torch.ByteTensor',
        'torch.CharTensor',
        'torch.ShortTensor',
        'torch.IntTensor',
        'torch.FloatTensor',
        'torch.DoubleTensor',
        'torch.LongTensor',
    }

    for k,t in ipairs(types) do

        -- [res] torch.cmul([res,] tensor1, tensor2)
        -- contiguous
        local m1 = torch.randn(10, 10, 10):type(t)
        local m2 = torch.randn(10, 10 * 10):type(t)
        local sm1 = m1[{4, {}, {}}]
        local sm2 = m2[{4, {}}]
        local res1 = torch.cmul(sm1, sm2)
        local res2 = res1:clone():zero()
        for i = 1,sm1:size(1) do
            for j = 1, sm1:size(2) do
                local idx1d = (((i-1)*sm1:size(1)))+j
                res2[i][j] = sm1[i][j] * sm2[idx1d]
            end
        end
        local err = res1:clone():zero()
        -- find absolute error
        for i = 1, res1:size(1) do
            for j = 1, res1:size(2) do
                err[i][j] = math.abs(res1[i][j] - res2[i][j])
            end
        end
        -- find maximum element of error
        local maxerr = 0
        for i = 1, err:size(1) do
            for j = 1, err:size(2) do
                if err[i][j] > maxerr then
                    maxerr = err[i][j]
                end
            end
        end
        mytester:assertlt(maxerr, precision, 'error in torch.cmul - contiguous' .. ' ' .. t)

        -- non-contiguous
        local m1 = torch.randn(10, 10, 10):type(t)
        local m2 = torch.randn(10 * 10, 10 * 10):type(t)
        local sm1 = m1[{{}, 4, {}}]
        local sm2 = m2[{{}, 4}]
        local res1 = torch.cmul(sm1, sm2)
        local res2 = res1:clone():zero()
        for i = 1,sm1:size(1) do
            for j = 1, sm1:size(2) do
                local idx1d = (((i-1)*sm1:size(1)))+j
                res2[i][j] = sm1[i][j] * sm2[idx1d]
            end
        end
        local err = res1:clone():zero()
        -- find absolute error
        for i = 1, res1:size(1) do
            for j = 1, res1:size(2) do
                err[i][j] = math.abs(res1[i][j] - res2[i][j])
            end
        end
        -- find maximum element of error
        local maxerr = 0
        for i = 1, err:size(1) do
            for j = 1, err:size(2) do
                if err[i][j] > maxerr then
                    maxerr = err[i][j]
                end
            end
        end
        mytester:assertlt(maxerr, precision, 'error in torch.cmul - non-contiguous' .. ' ' .. t)
    end
end

function torchtest.cpow()  -- [res] torch.cpow([res,] tensor1, tensor2)
   -- contiguous
   local m1 = torch.rand(10, 10, 10)
   local m2 = torch.rand(10, 10 * 10)
   local sm1 = m1[{4, {}, {}}]
   local sm2 = m2[{4, {}}]
   local res1 = torch.cpow(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
         local idx1d = (((i-1)*sm1:size(1)))+j
         res2[i][j] = math.pow(sm1[i][j], sm2[idx1d])
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
         err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
         if err[i][j] > maxerr then
            maxerr = err[i][j]
         end
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.cpow - contiguous')

   -- non-contiguous
   local m1 = torch.rand(10, 10, 10)
   local m2 = torch.rand(10 * 10, 10 * 10)
   local sm1 = m1[{{}, 4, {}}]
   local sm2 = m2[{{}, 4}]
   local res1 = torch.cpow(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
         local idx1d = (((i-1)*sm1:size(1)))+j
         res2[i][j] = math.pow(sm1[i][j],sm2[idx1d])
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
         err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
         if err[i][j] > maxerr then
            maxerr = err[i][j]
         end
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.cpow - non-contiguous')
end

function torchtest.sum()
   local x = torch.rand(msize,msize)
   local mx = torch.sum(x,2)
   local mxx = torch.Tensor()
   torch.sum(mxx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sum value')

   local y = torch.rand(5, 5, 5)
   for i=1,3 do
      local a = y:sum(i)
      local b = y:narrow(i, 1, 1):clone():zero()
      for j = 1, 5 do
         b:add(y:narrow(i, j, 1))
      end
      mytester:asserteq(maxdiff(a, b), 0, 'torch.sum value')
   end
end
function torchtest.prod()
   local x = torch.rand(msize,msize)
   local mx = torch.prod(x,2)
   local mxx = torch.Tensor()
   torch.prod(mxx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.prod value')

   local y = torch.rand(5, 5, 5)
   for i=1,3 do
      local a = y:prod(i)
      local b = y:narrow(i, 1, 1):clone():fill(1)
      for j = 1, 5 do
         b:cmul(y:narrow(i, j, 1))
      end
      mytester:asserteq(maxdiff(a, b), 0, 'torch.sum value')
   end
end
function torchtest.cumsum()
   local x = torch.rand(msize,msize)
   local mx = torch.cumsum(x,2)
   local mxx = torch.Tensor()
   torch.cumsum(mxx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cumsum value')
end
function torchtest.cumprod()
   local x = torch.rand(msize,msize)
   local mx = torch.cumprod(x,2)
   local mxx = torch.Tensor()
   torch.cumprod(mxx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cumprod value')
end
function torchtest.cross()
   local x = torch.rand(msize,3,msize)
   local y = torch.rand(msize,3,msize)
   local mx = torch.cross(x,y)
   local mxx = torch.Tensor()
   torch.cross(mxx,x,y)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cross value')
end
function torchtest.zeros()
   local mx = torch.zeros(msize,msize)
   local mxx = torch.Tensor()
   torch.zeros(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.zeros value')
end
function torchtest.histc()
   local x = torch.Tensor{ 2, 4, 2, 2, 5, 4 }
   local y = torch.histc(x, 5, 1, 5) -- nbins, min, max
   local z = torch.Tensor{ 0, 3, 0, 2, 1 }
   mytester:assertTensorEq(y,z,precision,'error in torch.histc')
end
function torchtest.bhistc()
   local x = torch.Tensor(3, 6)
   x[1] = torch.Tensor{ 2, 4, 2, 2, 5, 4 }
   x[2] = torch.Tensor{ 3, 5, 1, 5, 3, 5 }
   x[3] = torch.Tensor{ 3, 4, 2, 5, 5, 1 }
   local y = torch.bhistc(x, 5, 1, 5) -- nbins, min, max
   local z = torch.Tensor(3, 5)
   z[1] = torch.Tensor{ 0, 3, 0, 2, 1 }
   z[2] = torch.Tensor{ 1, 0, 2, 0, 3 }
   z[3] = torch.Tensor{ 1, 1, 1, 1, 2 }
   mytester:assertTensorEq(y,z,precision,'error in torch.bhistc in last dimension')
end
function torchtest.ones()
   local mx = torch.ones(msize,msize)
   local mxx = torch.Tensor()
   torch.ones(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.ones value')
end
function torchtest.diag()
   local x = torch.rand(msize,msize)
   local mx = torch.diag(x)
   local mxx = torch.Tensor()
   torch.diag(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.diag value')
end
function torchtest.eye()
   local mx = torch.eye(msize,msize)
   local mxx = torch.Tensor()
   torch.eye(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.eye value')
end
function torchtest.renorm()
   local m1 = torch.randn(10,5)
   local res1 = torch.Tensor()
   local m2

   local function renorm(matrix, value, dim, max_norm)
      local m1 = matrix:transpose(dim, 1):contiguous()
      -- collapse non-dim dimensions:
      m2 = m1:reshape(m1:size(1), m1:nElement()/m1:size(1))
      local norms = m2:norm(value,2)
      -- clip
      local new_norms = norms:clone()
      new_norms[torch.gt(norms, max_norm)] = max_norm
      new_norms:cdiv(norms:add(1e-7))
      -- renormalize
      m1:cmul(new_norms:expandAs(m1))
      return m1:transpose(dim, 1)
   end

   -- note that the axis fed to torch.renorm is different (2~=1)
   local maxnorm = m1:norm(2,1):mean()
   m2 = renorm(m1,2,2,maxnorm)

   m1:renorm(2,2,maxnorm)
   mytester:assertTensorEq(m1, m2, 0.00001, 'error in renorm')
   mytester:assertTensorEq(m1:norm(2,1), m2:norm(2,1), 0.00001, 'error in renorm')

   m1 = torch.randn(3,4,5)
   m2 = m1:transpose(2,3):contiguous():reshape(15,4)

   maxnorm = m2:norm(2,1):mean()
   m2 = renorm(m2,2,2,maxnorm)

   m1:renorm(2,2,maxnorm)
   local m3 = m1:transpose(2,3):contiguous():reshape(15,4)
   mytester:assertTensorEq(m3, m2, 0.00001, 'error in renorm')
   mytester:assertTensorEq(m3:norm(2,1), m2:norm(2,1), 0.00001, 'error in renorm')
end
function torchtest.multinomialwithreplacement()
   local n_row = 3
   for n_col=4,5 do
      local t=os.time()
      torch.manualSeed(t)
      local prob_dist = torch.rand(n_row,n_col)
      prob_dist:select(2,n_col):fill(0) --index n_col shouldn't be sampled
      local n_sample = n_col
      local sample_indices = torch.multinomial(prob_dist, n_sample, true)
      mytester:assert(prob_dist:dim() == 2, "wrong number of prob_dist dimensions")
      mytester:assert(sample_indices:size(2) == n_sample, "wrong number of samples")
      for i=1,n_row do
         for j=1,n_sample do
            mytester:assert(sample_indices[{i,j}] ~= n_col, "sampled an index with zero probability")
         end
      end
   end
end
function torchtest.multinomialwithoutreplacement()
   local n_row = 3
   for n_col=4,5 do
      local t=os.time()
      torch.manualSeed(t)
      local prob_dist = torch.rand(n_row,n_col)
      prob_dist:select(2,n_col):fill(0) --index n_col shouldn't be sampled
      local n_sample = 3
      local sample_indices = torch.multinomial(prob_dist, n_sample, false)
      mytester:assert(prob_dist:dim() == 2, "wrong number of prob_dist dimensions")
      mytester:assert(sample_indices:size(2) == n_sample, "wrong number of samples")
      for i=1,n_row do
         local row_samples = {}
         for j=1,n_sample do
            local sample_idx = sample_indices[{i,j}]
            mytester:assert(
                sample_idx ~= n_col, "sampled an index with zero probability"
            )
            mytester:assert(
                not row_samples[sample_idx], "sampled an index twice"
            )
            row_samples[sample_idx] = true
         end
      end
   end
end
function torchtest.aliasMultinomial()
   for i =1,5 do
      local n_class = 5
      local t=os.time()
      torch.manualSeed(t)
      local probs = torch.Tensor(n_class):uniform(0,1)
      probs:div(probs:sum())
      local output = torch.LongTensor(1000, 10000)
      local n_samples = output:nElement()
      local prob_state = torch.multinomialAliasSetup(probs)
      mytester:assert(prob_state[1]:min() > 0, "Index ="..prob_state[1]:min().."alias indices has an index below or equal to 0")
      mytester:assert(prob_state[1]:max() <= n_class, prob_state[1]:max().." alias indices has an index exceeding num_class")
      local prob_state = torch.multinomialAliasSetup(probs, prob_state)
      mytester:assert(prob_state[1]:min() > 0, "Index ="..prob_state[1]:min().."alias indices has an index below or equal to 0(cold)")
      mytester:assert(prob_state[1]:max() <= n_class, prob_state[1]:max()..","..prob_state[1]:min().." alias indices has an index exceeding num_class(cold)")
      local output = torch.LongTensor(n_samples)
      output = torch.multinomialAlias(output, prob_state)
      mytester:assert(output:nElement() == n_samples, "wrong number of samples")
      mytester:assert(output:min() > 0, "sampled indices has an index below or equal to 0")
      mytester:assert(output:max() <= n_class, "indices has an index exceeding num_class")
   end

end
function torchtest.multinomialvector()
   local n_col = 4
   local t=os.time()
   torch.manualSeed(t)
   local prob_dist = torch.rand(n_col)
   local n_sample = n_col
   local sample_indices = torch.multinomial(prob_dist, n_sample, true)
   local s_dim = sample_indices:dim()
   mytester:assert(s_dim == 1, "wrong number of returned dimensions: "..s_dim)
   mytester:assert(prob_dist:dim() == 1, "wrong number of prob_dist dimensions")
   mytester:assert(sample_indices:size(1) == n_sample, "wrong number of samples")
end
function torchtest.range()
   local mx = torch.range(0,1)
   local mxx = torch.Tensor()
   torch.range(mxx,0,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.range value')

   -- Check range for non-contiguous tensors.
   local x = torch.zeros(2, 3)
   local y = x:narrow(2, 2, 2)
   y:range(0, 3)
   mytester:assertTensorEq(x, torch.Tensor{{0, 0, 1}, {0, 2, 3}}, 1e-16,
                           'non-contiguous range failed')
end
function torchtest.rangenegative()
   local mx = torch.Tensor({1,0})
   local mxx = torch.Tensor()
   torch.range(mxx,1,0,-1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.range value for negative step')
end
function torchtest.rangeequalbounds()
   local mx = torch.Tensor({1})
   local mxx = torch.Tensor()
   torch.range(mxx,1,1,-1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.range value for equal bounds step')
   torch.range(mxx,1,1,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.range value for equal bounds step')
end
function torchtest.rangefloat()
   local mx = torch.FloatTensor():range(0.6, 0.9, 0.1)
   mytester:asserteq(mx:size(1), 4, 'wrong size for FloatTensor range')
   mx = torch.FloatTensor():range(1, 10, 0.3)
   mytester:asserteq(mx:size(1), 31, 'wrong size for FloatTensor range')
end
function torchtest.rangedouble()
   local mx = torch.DoubleTensor():range(0.6, 0.9, 0.1)
   mytester:asserteq(mx:size(1), 4, 'wrong size for DoubleTensor range')
   mx = torch.DoubleTensor():range(1, 10, 0.3)
   mytester:asserteq(mx:size(1), 31, 'wrong size for DoubleTensor range')
end
function torchtest.randperm()
   local t=os.time()
   torch.manualSeed(t)
   local mx = torch.randperm(msize)
   local mxx = torch.Tensor()
   torch.manualSeed(t)
   torch.randperm(mxx,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.randperm value')
end
function torchtest.reshape()
   local x = torch.rand(10,13,23)
   local mx = torch.reshape(x,130,23)
   local mxx = torch.Tensor()
   torch.reshape(mxx,x,130,23)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.reshape value')
end

local function assertIsOrdered(order, x, mxx, ixx, task)
  local areOrdered
  if order == 'descending' then
    areOrdered = function(a, b) return a >= b end
  elseif order == 'ascending' then
    areOrdered = function(a, b) return a <= b end
  else
    error('unknown order "' .. order .. '", must be "ascending" or "descending"')
  end

  local decreasing = true
  for j = 1,msize do
    for k = 2,msize do
      decreasing = decreasing and areOrdered(mxx[j][k-1], mxx[j][k])
    end
  end
  mytester:assert(decreasing, 'torch.sort (' .. order .. ') values unordered for ' .. task)
  local seen = torch.ByteTensor(msize)
  local indicesCorrect = true
  for k = 1,msize do
    seen:zero()
    for j = 1,msize do
      indicesCorrect = indicesCorrect and (x[k][ixx[k][j]] == mxx[k][j])
      seen[ixx[k][j]] = 1
    end
    indicesCorrect = indicesCorrect and (torch.sum(seen) == msize)
  end
  mytester:assert(indicesCorrect, 'torch.sort (' .. order .. ') indices wrong for ' .. task)
end

function torchtest.sortAscending()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.sort(x)

   -- Test use of result tensor
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.sort(mxx,ixx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sort (ascending) value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.sort (ascending) index')

   -- Test sorting of random numbers
   assertIsOrdered('ascending', x, mxx, ixx, 'random')

   mytester:assertTensorEq(
           torch.sort(torch.Tensor{ 50, 40, 30, 20, 10 }),
           torch.Tensor{ 10, 20, 30, 40, 50 },
           1e-16,
           "torch.sort (ascending) simple sort"
       )
   -- Test that we still have proper sorting with duplicate keys
   local x = torch.floor(torch.rand(msize,msize)*10)
   torch.sort(mxx,ixx,x)
   assertIsOrdered('ascending', x, mxx, ixx, 'random with duplicate keys')
end

function torchtest.sortDescending()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.sort(x,true)

   -- Test use of result tensor
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.sort(mxx,ixx,x,true)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sort (descending) value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.sort (descending) index')

   -- Test sorting of random numbers
   assertIsOrdered('descending', x, mxx, ixx, 'random')

   -- Test simple sort task
   mytester:assertTensorEq(
           torch.sort(torch.Tensor{ 10, 20, 30, 40, 50 },true),
           torch.Tensor{ 50, 40, 30, 20, 10 },
           1e-16,
           "torch.sort (descending) simple sort"
       )

   -- Test that we still have proper sorting with duplicate keys
   assertIsOrdered('descending', x, mxx, ixx, 'random with duplicate keys')
end

function torchtest.topK()
   local function topKViaSort(t, k, dim, dir)
      local sorted, indices = t:sort(dim, dir)
      return sorted:narrow(dim, 1, k), indices:narrow(dim, 1, k)
   end

   local function compareTensors(t, res1, ind1, res2, ind2, dim, msg)
      -- Values should be exactly equivalent
      mytester:assertTensorEq(res1, res2, 0, msg)

      -- Indices might differ based on the implementation, since there is
      -- no guarantee of the relative order of selection
      if ind1:eq(ind2):min() == 0 then
         -- To verify that the indices represent equivalent elements,
         -- gather from the input using the topk indices and compare against
         -- the sort indices
         local vals = t:gather(dim, ind2)
         mytester:assertTensorEq(res1, vals, 0, msg)
      end
   end

   local function compare(t, k, dim, dir, msg)
      local topKVal, topKInd = t:topk(k, dim, dir, true)
      local sortKVal, sortKInd = topKViaSort(t, k, dim, dir)

      compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim, msg)
   end

   local t = torch.rand(math.random(1, msize),
                        math.random(1, msize),
                        math.random(1, msize))

   for kTries = 1, 3 do
      for dimTries = 1, 3 do
         for _, transpose in ipairs({true, false}) do
            for _, dir in ipairs({true, false}) do
               local testTensor = t

               local transposeMsg = nil
               if transpose then
                  local dim1 = math.random(1, t:nDimension())
                  local dim2 = dim1

                  while dim1 == dim2 do
                     dim2 = math.random(1, t:nDimension())
                  end

                  testTensor = t:transpose(dim1, dim2)
                  transposeMsg = 'transpose(' .. dim1 .. ', ' .. dim2 .. ')'
               end

               local dim = math.random(1, testTensor:nDimension())
               local k = math.random(1, testTensor:size(dim))
               local msg = 'topk(' .. k .. ', ' .. dim .. ', ' .. tostring(dir) .. ', true)'
               if transposeMsg then
                  msg = msg .. ' ' .. transposeMsg
               end

               compare(testTensor, k, dim, dir, msg)
            end
         end
      end
   end
end

function torchtest.kthvalue()
   local x = torch.rand(msize, msize, msize)
   local x0 = x:clone()
   do
      local k = math.random(1, msize)
      local mx, ix = torch.kthvalue(x, k)
      local mxx, ixx = torch.sort(x)

      mytester:assertTensorEq(mxx:select(3, k), mx:select(3, 1), 0,
                              'torch.kthvalue value')
      mytester:assertTensorEq(ixx:select(3, k), ix:select(3, 1), 0,
                              'torch.kthvalue index')
   end
   do -- test use of result tensors
      local k = math.random(1, msize)
      local mx = torch.Tensor()
      local ix = torch.LongTensor()
      torch.kthvalue(mx, ix, x, k)
      local mxx, ixx = torch.sort(x)
      mytester:assertTensorEq(mxx:select(3, k), mx:select(3, 1), 0,
                              'torch.kthvalue value')
      mytester:assertTensorEq(ixx:select(3, k), ix:select(3, 1), 0,
                              'torch.kthvalue index')
   end
   do -- test non-default dim
      local k = math.random(1, msize)
      local mx, ix = torch.kthvalue(x, k, 1)
      local mxx, ixx = torch.sort(x, 1)
      mytester:assertTensorEq(mxx:select(1, k), mx[1], 0,
                              'torch.kthvalue value')
      mytester:assertTensorEq(ixx:select(1, k), ix[1], 0,
                              'torch.kthvalue index')
   end
   do -- non-contiguous
      local y = x:narrow(2, 1, 1)
      local y0 = y:clone()
      local k = math.random(1, msize)
      local my, ix = torch.kthvalue(y, k)
      local my0, ix0 = torch.kthvalue(y0, k)
      mytester:assertTensorEq(my, my0, 0, 'torch.kthvalue value')
      mytester:assertTensorEq(ix, ix0, 0, 'torch.kthvalue index')
   end
   mytester:assertTensorEq(x, x0, 0, 'torch.kthvalue modified input')

   -- simple test case (with repetitions)
   local y = torch.Tensor{3,5,4,1,1,5}
   mytester:assertTensorEq(torch.kthvalue(y, 3), torch.Tensor{3}, 1e-16,
      'torch.kthvalue simple')
   mytester:assertTensorEq(torch.kthvalue(y, 2), torch.Tensor{1}, 1e-16,
      'torch.kthvalue simple')
end

function torchtest.median()
   for _, msize in ipairs{155,156} do
      local x = torch.rand(msize, msize)
      local x0 = x:clone()

      local mx, ix = torch.median(x)
      local mxx, ixx = torch.sort(x)
      local ind = math.floor((msize+1)/2)

      mytester:assertTensorEq(mxx:select(2, ind), mx:select(2, 1), 0,
                              'torch.median value')
      mytester:assertTensorEq(ixx:select(2, ind), ix:select(2, 1), 0,
                              'torch.median index')

      -- Test use of result tensor
      local mr = torch.Tensor()
      local ir = torch.LongTensor()
      torch.median(mr, ir, x)
      mytester:assertTensorEq(mr, mx, 0, 'torch.median result tensor value')
      mytester:assertTensorEq(ir, ix, 0, 'torch.median result tensor index')

      -- Test non-default dim
      mx, ix = torch.median(x, 1)
      mxx, ixx = torch.sort(x, 1)
      mytester:assertTensorEq(mxx:select(1, ind), mx[1], 0,
                              'torch.median value')
      mytester:assertTensorEq(ixx:select(1, ind), ix[1], 0,
                              'torch.median index')

      -- input unchanged
      mytester:assertTensorEq(x, x0, 0, 'torch.median modified input')
   end
end

function torchtest.mode()
   local x = torch.range(1, msize * msize):reshape(msize, msize)
   x:select(1, 1):fill(1)
   x:select(1, 2):fill(1)
   x:select(2, 1):fill(1)
   x:select(2, 2):fill(1)
   local x0 = x:clone()

   -- Pre-calculated results.
   local res = torch.Tensor(msize):fill(1)
   -- The indices are the position of the last appearance of the mode element.
   local resix = torch.LongTensor(msize):fill(2)
   resix[1] = msize
   resix[2] = msize

   local mx, ix = torch.mode(x)

   mytester:assertTensorEq(res:view(msize, 1), mx, 0, 'torch.mode value')
   mytester:assertTensorEq(resix:view(msize, 1), ix, 0, 'torch.mode index')

   -- Test use of result tensor
   local mr = torch.Tensor()
   local ir = torch.LongTensor()
   torch.mode(mr, ir, x)
   mytester:assertTensorEq(mr, mx, 0, 'torch.mode result tensor value')
   mytester:assertTensorEq(ir, ix, 0, 'torch.mode result tensor index')

   -- Test non-default dim
   mx, ix = torch.mode(x, 1)
   mytester:assertTensorEq(res:view(1, msize), mx, 0, 'torch.mode value')
   mytester:assertTensorEq(resix:view(1, msize), ix, 0, 'torch.mode index')

   local input = torch.Tensor({
       {1, 2, 2, 2, 3, 2},
       {1.5, 2, 2, 1.5, 1.5, 5},
   })
   local value, index = torch.mode(input)
   local expected_value = torch.Tensor({{2}, {1.5}})
   mytester:assertTensorEq(value, expected_value)

   -- input unchanged
   mytester:assertTensorEq(x, x0, 0, 'torch.mode modified input')
end


function torchtest.tril()
   local x = torch.rand(msize,msize)
   local mx = torch.tril(x)
   local mxx = torch.Tensor()
   torch.tril(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.tril value')
end
function torchtest.triu()
   local x = torch.rand(msize,msize)
   local mx = torch.triu(x)
   local mxx = torch.Tensor()
   torch.triu(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.tril value')
end
function torchtest.cat()
   for dim = 1, 3 do
      local x = torch.rand(13, msize, msize):transpose(1, dim)
      local y = torch.rand(17, msize, msize):transpose(1, dim)
      local mx = torch.cat(x, y, dim)
      mytester:assertTensorEq(mx:narrow(dim, 1, 13), x, 0, 'torch.cat value')
      mytester:assertTensorEq(mx:narrow(dim, 14, 17), y, 0, 'torch.cat value')

      local mxx = torch.Tensor()
      torch.cat(mxx, x, y, dim)
      mytester:assertTensorEq(mx, mxx, 0, 'torch.cat value')

      local x = torch.rand(1,2,3)
      local y = torch.Tensor()
      local mx = torch.cat(x,y,dim)
      mytester:asserteq(mx:size(1),1,'torch.cat size')
      mytester:asserteq(mx:size(2),2,'torch.cat size')
      mytester:asserteq(mx:size(3),3,'torch.cat size')
      mytester:assertTensorEq(mx, x, 0, 'torch.cat value')

      local x = torch.Tensor()
      local y = torch.Tensor()
      local mx = torch.cat(x,y,dim)
      mytester:asserteq(mx:dim(),0,'torch.cat dim')
   end
   local x = torch.Tensor()
   local y = torch.rand(1,2,3)
   local mx = torch.cat(x,y)
   mytester:asserteq(mx:size(1),1,'torch.cat size')
   mytester:asserteq(mx:size(2),2,'torch.cat size')
   mytester:asserteq(mx:size(3),3,'torch.cat size')
   mytester:assertTensorEq(mx, y, 0, 'torch.cat value')

   local x = torch.Tensor()
   local y = torch.Tensor()
   local mx = torch.cat(x,y)
   mytester:asserteq(mx:dim(),0,'torch.cat dim')
end
function torchtest.catArray()
   for dim = 1, 3 do
      local x = torch.rand(13, msize, msize):transpose(1, dim)
      local y = torch.rand(17, msize, msize):transpose(1, dim)
      local z = torch.rand(19, msize, msize):transpose(1, dim)

      local mx = torch.cat({x, y, z}, dim)
      mytester:assertTensorEq(mx:narrow(dim, 1, 13), x, 0, 'torch.cat value')
      mytester:assertTensorEq(mx:narrow(dim, 14, 17), y, 0, 'torch.cat value')
      mytester:assertTensorEq(mx:narrow(dim, 31, 19), z, 0, 'torch.cat value')

      mytester:assertError(function() torch.cat{} end, 'torch.cat empty table')

      local mxx = torch.Tensor()
      torch.cat(mxx, {x, y, z}, dim)
      mytester:assertTensorEq(mx, mxx, 0, 'torch.cat value')
      torch.cat(mxx:float(), {x:float(), y:float(), z:float()}, dim)
      mytester:assertTensorEq(mx, mxx, 0, 'torch.cat value')
      torch.cat(mxx:double(), {x:double(), y:double(), z:double()}, dim)
      mytester:assertTensorEq(mx, mxx, 0, 'torch.cat value')

      local x = torch.rand(1,2,3)
      local y = torch.Tensor()
      local mx = torch.cat({x,y},dim)
      mytester:asserteq(mx:size(1),1,'torch.cat size')
      mytester:asserteq(mx:size(2),2,'torch.cat size')
      mytester:asserteq(mx:size(3),3,'torch.cat size')
      mytester:assertTensorEq(mx, x, 0, 'torch.cat value')

      local x = torch.Tensor()
      local y = torch.Tensor()
      local mx = torch.cat({x,y},dim)
      mytester:asserteq(mx:dim(),0,'torch.cat dim')
   end
   local x = torch.Tensor()
   local y = torch.rand(1,2,3)
   local mx = torch.cat({x,y})
   mytester:asserteq(mx:size(1),1,'torch.cat size')
   mytester:asserteq(mx:size(2),2,'torch.cat size')
   mytester:asserteq(mx:size(3),3,'torch.cat size')
   mytester:assertTensorEq(mx, y, 0, 'torch.cat value')

   local x = torch.Tensor()
   local y = torch.Tensor()
   local mx = torch.cat({x,y})
   mytester:asserteq(mx:dim(),0,'torch.cat dim')
end
function torchtest.catNoDim()
   local a
   local b
   local c

   a = torch.Tensor(msize):uniform()
   b = torch.Tensor(msize):uniform()
   c = torch.cat(a, b)
   mytester:assertTensorEq(c:narrow(1, 1, msize), a, 0, 'torch.cat value')
   mytester:assertTensorEq(c:narrow(1, msize + 1, msize), b, 0, 'torch.cat value')

   a = torch.Tensor(1, msize):uniform()
   b = torch.Tensor(1, msize):uniform()
   c = torch.cat(a, b)
   mytester:assertTensorEq(c:narrow(2, 1, msize), a, 0, 'torch.cat value')
   mytester:assertTensorEq(c:narrow(2, msize + 1, msize), b, 0, 'torch.cat value')

   a = torch.Tensor(10, msize):uniform()
   b = torch.Tensor(10, msize):uniform()
   c = torch.cat(a, b)
   mytester:assertTensorEq(c:narrow(2, 1, msize), a, 0, 'torch.cat value')
   mytester:assertTensorEq(c:narrow(2, msize + 1, msize), b, 0, 'torch.cat value')
end
function torchtest.sin_2()
   local x = torch.rand(msize,msize,msize)
   local mx = torch.sin(x)
   local mxx  = torch.Tensor()
   torch.sin(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sin value')
end
function torchtest.linspace()
   local from = math.random()
   local to = from+math.random()
   local mx = torch.linspace(from,to,137)
   local mxx = torch.Tensor()
   torch.linspace(mxx,from,to,137)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.linspace value')
   mytester:assertError(function() torch.linspace(0,1,1) end, 'accepted 1 point between 2 distinct endpoints')
   mytester:assertTensorEq(torch.linspace(0,0,1),torch.zeros(1),1e-16, 'failed to generate for torch.linspace(0,0,1)')

   -- Check linspace for generating with start > end.
   mytester:assertTensorEq(torch.linspace(2,0,3),
                           torch.Tensor{2,1,0},
                           1e-16,
                           'failed to generate for torch.linspace(2,0,3)')

   -- Check linspace for non-contiguous tensors.
   local x = torch.zeros(2, 3)
   local y = x:narrow(2, 2, 2)
   y:linspace(0, 3, 4)
   mytester:assertTensorEq(x, torch.Tensor{{0, 0, 1}, {0, 2, 3}}, 1e-16,
                           'non-contiguous linspace failed')
end
function torchtest.logspace()
   local from = math.random()
   local to = from+math.random()
   local mx = torch.logspace(from,to,137)
   local mxx = torch.Tensor()
   torch.logspace(mxx,from,to,137)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.logspace value')
   mytester:assertError(function() torch.logspace(0,1,1) end, 'accepted 1 point between 2 distinct endpoints')
   mytester:assertTensorEq(torch.logspace(0,0,1),torch.ones(1),1e-16, 'failed to generate for torch.linspace(0,0,1)')

   -- Check logspace for generating with start > end.
   mytester:assertTensorEq(torch.logspace(1,0,2),
                           torch.Tensor{10, 1},
                           1e-16,
                           'failed to generate for torch.logspace(1,0,2)')

   -- Check logspace for non-contiguous tensors.
   local x = torch.zeros(2, 3)
   local y = x:narrow(2, 2, 2)
   y:logspace(0, 3, 4)
   mytester:assertTensorEq(x, torch.Tensor{{0, 1, 10}, {0, 100, 1000}}, 1e-16,
                           'non-contiguous logspace failed')
end
function torchtest.rand()
   torch.manualSeed(123456)
   local mx = torch.rand(msize,msize)
   local mxx = torch.Tensor()
   torch.manualSeed(123456)
   torch.rand(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.rand value')
end
function torchtest.randn()
   torch.manualSeed(123456)
   local mx = torch.randn(msize,msize)
   local mxx = torch.Tensor()
   torch.manualSeed(123456)
   torch.randn(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.randn value')
end
function torchtest.gesv()
   if not torch.gesv then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                         {-6.05, -3.30,  5.36, -4.44,  1.08},
                         {-0.45,  2.58, -2.70,  0.27,  9.04},
                         {8.32,  2.71,  4.35, -7.17,  2.14},
                         {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()
   local b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                         {-1.56,  4.00, -8.67,  1.75,  2.86},
                         {9.81, -4.09, -4.57, -8.61,  8.99}}):t()
   local mx = torch.gesv(b,a)
   mytester:assertlt(b:dist(a*mx),1e-12,'torch.gesv')
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gesv(tb,ta,b,a)
   local mxxx = torch.gesv(b,a,b,a)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gesv value temp')
   mytester:asserteq(maxdiff(mx,b),0,'torch.gesv value flag')
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.gesv value out1')
   mytester:asserteq(maxdiff(mx,mxxx),0,'torch.gesv value out2')
end
function torchtest.gesv_reuse()
   if not torch.gesv then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                         {-6.05, -3.30,  5.36, -4.44,  1.08},
                         {-0.45,  2.58, -2.70,  0.27,  9.04},
                         {8.32,  2.71,  4.35, -7.17,  2.14},
                         {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()
   local b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                         {-1.56,  4.00, -8.67,  1.75,  2.86},
                         {9.81, -4.09, -4.57, -8.61,  8.99}}):t()
   local mx = torch.gesv(b,a)
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   torch.gesv(tb,ta,b,a)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gesv value temp')
   torch.gesv(tb,ta,b,a)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gesv value reuse')
end
function torchtest.trtrs()
   if not torch.trtrs then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                         {-6.05, -3.30,  5.36, -4.44,  1.08},
                         {-0.45,  2.58, -2.70,  0.27,  9.04},
                         {8.32,  2.71,  4.35, -7.17,  2.14},
                         {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()
   local b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                         {-1.56,  4.00, -8.67,  1.75,  2.86},
                         {9.81, -4.09, -4.57, -8.61,  8.99}}):t()

   local U = torch.triu(a)
   local L = torch.tril(a)

   -- solve Ux = b
   local x = torch.trtrs(b, U)
   mytester:assertlt(b:dist(U*x),1e-12,'torch.trtrs')
   x = torch.trtrs(b, U, 'U', 'N', 'N')
   mytester:assertlt(b:dist(U*x),1e-12,'torch.trtrs')

   -- solve Lx = b
   x = torch.trtrs(b, L, 'L')
   mytester:assertlt(b:dist(L*x),1e-12,'torch.trtrs')
   x = torch.trtrs(b, L, 'L', 'N', 'N')
   mytester:assertlt(b:dist(L*x),1e-12,'torch.trtrs')

   -- solve U'x = b
   x = torch.trtrs(b, U, 'U', 'T')
   mytester:assertlt(b:dist(U:t()*x),1e-12,'torch.trtrs')
   x = torch.trtrs(b, U, 'U', 'T', 'N')
   mytester:assertlt(b:dist(U:t()*x),1e-12,'torch.trtrs')

   -- solve U'x = b by manual transposition
   y = torch.trtrs(b, U:t(), 'L', 'N')
   mytester:assertlt(x:dist(y),1e-12,'torch.trtrs')

   -- solve L'x = b
   x = torch.trtrs(b, L, 'L', 'T')
   mytester:assertlt(b:dist(L:t()*x),1e-12,'torch.trtrs')
   x = torch.trtrs(b, L, 'L', 'T', 'N')
   mytester:assertlt(b:dist(L:t()*x),1e-12,'torch.trtrs')

   -- solve L'x = b by manual transposition
   y = torch.trtrs(b, L:t(), 'U', 'N')
   mytester:assertlt(x:dist(y),1e-12,'torch.trtrs')
end
function torchtest.trtrs_reuse()
   if not torch.trtrs then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                         {-6.05, -3.30,  5.36, -4.44,  1.08},
                         {-0.45,  2.58, -2.70,  0.27,  9.04},
                         {8.32,  2.71,  4.35, -7.17,  2.14},
                         {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()
   local b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                         {-1.56,  4.00, -8.67,  1.75,  2.86},
                         {9.81, -4.09, -4.57, -8.61,  8.99}}):t()
   local mx = torch.trtrs(b,a)
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   torch.trtrs(tb,ta,b,a)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.trtrs value temp')
   tb:zero()
   torch.trtrs(tb,ta,b,a)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.trtrs value reuse')
end
function torchtest.gels_uniquely_determined()
   if not torch.gels then return end
   local expectedNorm = 0
   local a=torch.Tensor({{ 1.44, -9.96, -7.55,  8.34},
                         {-7.84, -0.28,  3.24,  8.09},
                         {-4.39, -3.24,  6.27,  5.28},
                         {4.53,  3.83, -6.64,  2.06}}):t()
   local b=torch.Tensor({{8.58,  8.26,  8.48, -5.28},
                         {9.35, -4.43, -0.70, -0.26}}):t()
   local a_copy = a:clone()
   local b_copy = b:clone()
   local mx = torch.gels(b,a)
   mytester:asserteq(maxdiff(a,a_copy),0,'torch.gels changed a')
   mytester:asserteq(maxdiff(b,b_copy),0,'torch.gels changed b')
   mytester:assertalmosteq((torch.mm(a,mx)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')

   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gels(tb,ta,b,a)
   mytester:asserteq(maxdiff(a,a_copy),0,'torch.gels changed a')
   mytester:asserteq(maxdiff(b,b_copy),0,'torch.gels changed b')
   mytester:assertalmosteq((torch.mm(a,tb)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')

   local mxxx = torch.gels(b,a,b,a)
   mytester:assertalmosteq((torch.mm(a_copy,b)-b_copy):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gels value temp')
   mytester:asserteq(maxdiff(mx,b),0,'torch.gels value flag')
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.gels value out1')
   mytester:asserteq(maxdiff(mx,mxxx),0,'torch.gels value out2')
end
function torchtest.gels_reuse()
   if not torch.gels then return end
   local expectedNorm = 0
   local a=torch.Tensor({{ 1.44, -9.96, -7.55,  8.34},
                         {-7.84, -0.28,  3.24,  8.09},
                         {-4.39, -3.24,  6.27,  5.28},
                         {4.53,  3.83, -6.64,  2.06}}):t()
   local b=torch.Tensor({{8.58,  8.26,  8.48, -5.28},
                         {9.35, -4.43, -0.70, -0.26}}):t()
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   torch.gels(tb,ta,b,a)
   mytester:assertalmosteq((torch.mm(a,tb)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')
   torch.gels(tb,ta,b,a)
   mytester:assertalmosteq((torch.mm(a,tb)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')
   torch.gels(tb,ta,b,a)
   mytester:assertalmosteq((torch.mm(a,tb)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')
end
function torchtest.gels_overdetermined()
   if not torch.gels then return end
   local expectedNorm = 17.390200628863
   local a=torch.Tensor({{ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45},
                         {-7.84, -0.28,  3.24,  8.09,  2.52, -5.70},
                         {-4.39, -3.24,  6.27,  5.28,  0.74, -1.19},
                         {4.53,  3.83, -6.64,  2.06, -2.47,  4.70}}):t()
   local b=torch.Tensor({{8.58,  8.26,  8.48, -5.28,  5.72,  8.93},
                         {9.35, -4.43, -0.70, -0.26, -7.36, -2.52}}):t()
   local a_copy = a:clone()
   local b_copy = b:clone()
   local mx = torch.gels(b,a)
   mytester:asserteq(maxdiff(a,a_copy),0,'torch.gels changed a')
   mytester:asserteq(maxdiff(b,b_copy),0,'torch.gels changed b')
   mytester:assertalmosteq((torch.mm(a, mx)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')

   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gels(tb,ta,b,a)
   mytester:asserteq(maxdiff(a,a_copy),0,'torch.gels changed a')
   mytester:asserteq(maxdiff(b,b_copy),0,'torch.gels changed b')
   mytester:assertalmosteq((torch.mm(a,tb)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')

   local mxxx = torch.gels(b,a,b,a)
   mytester:assertalmosteq((torch.mm(a_copy,b)-b_copy):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gels value temp')
   mytester:asserteq(maxdiff(mx,b),0,'torch.gels value flag')
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.gels value out1')
   mytester:asserteq(maxdiff(mx,mxxx),0,'torch.gels value out2')
end
function torchtest.gels_underdetermined()
   if not torch.gels then return end
   local expectedNorm = 0
   local a=torch.Tensor({{ 1.44, -9.96, -7.55},
                         {-7.84, -0.28,  3.24},
                         {-4.39, -3.24,  6.27},
                         {4.53,  3.83, -6.64}}):t()
   local b=torch.Tensor({{8.58,  8.26,  8.48},
                         {9.35, -4.43, -0.70}}):t()

   local a_copy = a:clone()
   local b_copy = b:clone()
   local mx = torch.gels(b,a)
   mytester:asserteq(maxdiff(a,a_copy),0,'torch.gels changed a')
   mytester:asserteq(maxdiff(b,b_copy),0,'torch.gels changed b')
   mytester:assertalmosteq((torch.mm(a,mx)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')

   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gels(tb,ta,b,a)
   mytester:asserteq(maxdiff(a,a_copy),0,'torch.gels changed a')
   mytester:asserteq(maxdiff(b,b_copy),0,'torch.gels changed b')
   mytester:assertalmosteq((torch.mm(a,tb)-b):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')

   local mxxx = torch.gels(b,a,b,a)
   mytester:assertalmosteq((torch.mm(a_copy,b)-b_copy):norm(), expectedNorm, 1e-8, 'torch.gels wrong answer')
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gels value temp')
   mytester:asserteq(maxdiff(mx,b),0,'torch.gels value flag')
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.gels value out1')
   mytester:asserteq(maxdiff(mx,mxxx),0,'torch.gels value out2')
end
function torchtest.eig()
   if not torch.eig then return end
   local a=torch.Tensor({{ 1.96,  0.00,  0.00,  0.00,  0.00},
                         {-6.49,  3.80,  0.00,  0.00,  0.00},
                         {-0.47, -6.39,  4.17,  0.00,  0.00},
                         {-7.20,  1.50, -1.51,  5.70,  0.00},
                         {-0.65, -6.34,  2.67,  1.80, -7.10}}):t():clone()
   local e = torch.eig(a)
   local ee,vv = torch.eig(a,'V')
   local te = torch.Tensor()
   local tv = torch.Tensor()
   local eee,vvv = torch.eig(te,tv,a,'V')
   mytester:assertlt(maxdiff(e,ee),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(ee,eee),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(ee,te),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(vv,vvv),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(vv,tv),1e-12,'torch.eig value')
end
function torchtest.eig_reuse()
   if not torch.eig then return end
   local X = torch.randn(4,4)
   X = X:t()*X
   local e, v = torch.zeros(4,2), torch.zeros(4,4)
   torch.eig(e, v, X,'V')
   local Xhat = v * torch.diag(e:select(2,1)) * v:t()
   mytester:assertTensorEq(X, Xhat, 1e-8, 'VeV\' wrong')
   mytester:assert(not v:isContiguous(), 'V is contiguous')

   torch.eig(e, v, X, 'V')
   local Xhat = torch.mm(v, torch.mm(e:select(2,1):diag(), v:t()))
   mytester:assertTensorEq(X, Xhat, 1e-8, 'VeV\' wrong')
   mytester:assert(not v:isContiguous(), 'V is contiguous')
end
function torchtest.eig_noncontig()
   if not torch.eig then return end
   local X = torch.randn(4,4)
   X = X:t()*X
   local e = torch.zeros(4,2,2)[{ {}, 2, {} }]
   local v = torch.zeros(4,2,4)[{ {}, 2, {} }]
   mytester:assert(not v:isContiguous(), 'V is contiguous')
   mytester:assert(not e:isContiguous(), 'E is contiguous')
   torch.eig(e, v, X,'V')
   local Xhat = v * torch.diag(e:select(2,1)) * v:t()
   mytester:assertTensorEq(X, Xhat, 1e-8, 'VeV\' wrong')
end
function torchtest.test_symeig()
  if not torch.symeig then return end
  local xval = torch.rand(100,3)
  local cov = torch.mm(xval:t(), xval)
  local rese = torch.zeros(3)
  local resv = torch.zeros(3,3)

  -- First call to symeig
  mytester:assert(resv:isContiguous(), 'resv is not contiguous') -- PASS
  torch.symeig(rese, resv, cov:clone(), 'V')
  local ahat = resv*torch.diag(rese)*resv:t()
  mytester:assertTensorEq(cov, ahat, 1e-8, 'VeV\' wrong') -- PASS

  -- Second call to symeig
  mytester:assert(not resv:isContiguous(), 'resv is contiguous') -- FAIL
  torch.symeig(rese, resv, cov:clone(), 'V')
  local ahat = torch.mm(torch.mm(resv, torch.diag(rese)), resv:t())
  mytester:assertTensorEq(cov, ahat, 1e-8, 'VeV\' wrong') -- FAIL
end
function  torchtest.symeig_noncontig()
  if not torch.symeig then return end
   local X = torch.rand(5,5)
   X = X:t()*X
   local e = torch.zeros(4,2):select(2,2)
   local v = torch.zeros(4,2,4)[{ {}, 2, {} }]
   mytester:assert(not v:isContiguous(), 'V is contiguous')
   mytester:assert(not e:isContiguous(), 'E is contiguous')
   torch.symeig(e, v, X,'V')
   local Xhat = v * torch.diag(e) * v:t()
   mytester:assertTensorEq(X, Xhat, 1e-8, 'VeV\' wrong')
end
function torchtest.svd()
   if not torch.svd then return end
   local a=torch.Tensor({{8.79,  6.11, -9.15,  9.57, -3.49,  9.84},
                         {9.93,  6.91, -7.93,  1.64,  4.02,  0.15},
                         {9.83,  5.04,  4.86,  8.83,  9.80, -8.99},
                         {5.45, -0.27,  4.85,  0.74, 10.00, -6.02},
                         {3.16,  7.98,  3.01,  5.80,  4.27, -5.31}}):t():clone()
   local u,s,v = torch.svd(a)
   local uu = torch.Tensor()
   local ss = torch.Tensor()
   local vv = torch.Tensor()
   local uuu,sss,vvv = torch.svd(uu,ss,vv,a)
   mytester:asserteq(maxdiff(u,uu),0,'torch.svd')
   mytester:asserteq(maxdiff(u,uuu),0,'torch.svd')
   mytester:asserteq(maxdiff(s,ss),0,'torch.svd')
   mytester:asserteq(maxdiff(s,sss),0,'torch.svd')
   mytester:asserteq(maxdiff(v,vv),0,'torch.svd')
   mytester:asserteq(maxdiff(v,vvv),0,'torch.svd')
end
function torchtest.svd_reuse()
   if not torch.svd then return end
   local X = torch.randn(4,4)
   local U, S, V = torch.svd(X)
   local Xhat = torch.mm(U, torch.mm(S:diag(), V:t()))
   mytester:assertTensorEq(X, Xhat, 1e-8, 'USV\' wrong')

   mytester:assert(not U:isContiguous(), 'U is contiguous')
   torch.svd(U, S, V, X)
   local Xhat = torch.mm(U, torch.mm(S:diag(), V:t()))
   mytester:assertTensorEq(X, Xhat, 1e-8, 'USV\' wrong')
end
function torchtest.svd_noncontig()
   if not torch.svd then return end
   local X = torch.randn(5,5)
   local U = torch.zeros(5,2,5)[{ {}, 2, {} }]
   local S = torch.zeros(5,2)[{ {}, 2 }]
   local V = torch.zeros(5,2,5)[{ {}, 2, {} }]

   mytester:assert(not U:isContiguous(), 'U is contiguous')
   mytester:assert(not S:isContiguous(), 'S is contiguous')
   mytester:assert(not V:isContiguous(), 'V is contiguous')
   torch.svd(U, S, V, X)
   local Xhat = torch.mm(U, torch.mm(S:diag(), V:t()))
   mytester:assertTensorEq(X, Xhat, 1e-8, 'USV\' wrong')
end
function torchtest.inverse()
   if not torch.inverse then return end
   local M = torch.randn(5,5)
   local MI = torch.inverse(M)
   local E = torch.eye(5)
   mytester:assert(not MI:isContiguous(), 'MI is contiguous')
   mytester:assertalmosteq(maxdiff(E,torch.mm(M,MI)), 0, 1e-8, 'inverse value')
   mytester:assertalmosteq(maxdiff(E,torch.mm(MI,M)), 0, 1e-8, 'inverse value')

   local MII = torch.Tensor(5,5)
   torch.inverse(MII, M)
   mytester:assert(not MII:isContiguous(), 'MII is contiguous')
   mytester:asserteq(maxdiff(MII, MI), 0, 'inverse value in-place')
   -- second call, now that MII is transposed
   torch.inverse(MII, M)
   mytester:assert(not MII:isContiguous(), 'MII is contiguous')
   mytester:asserteq(maxdiff(MII, MI), 0, 'inverse value in-place')
end
function torchtest.conv2()
   local x = torch.rand(math.floor(torch.uniform(50,100)),math.floor(torch.uniform(50,100)))
   local k = torch.rand(math.floor(torch.uniform(10,20)),math.floor(torch.uniform(10,20)))
   local imvc = torch.conv2(x,k)
   local imvc2 = torch.conv2(x,k,'V')
   local imfc = torch.conv2(x,k,'F')

   local ki = k:clone();
   local ks = k:storage()
   local kis = ki:storage()
   for i=ks:size(),1,-1 do kis[ks:size()-i+1]=ks[i] end
   local imvx = torch.xcorr2(x,ki)
   local imvx2 = torch.xcorr2(x,ki,'V')
   local imfx = torch.xcorr2(x,ki,'F')

   mytester:asserteq(maxdiff(imvc,imvc2),0,'torch.conv2')
   mytester:asserteq(maxdiff(imvc,imvx),0,'torch.conv2')
   mytester:asserteq(maxdiff(imvc,imvx2),0,'torch.conv2')
   mytester:asserteq(maxdiff(imfc,imfx),0,'torch.conv2')
   mytester:assertlt(math.abs(x:dot(x)-torch.xcorr2(x,x)[1][1]),1e-10,'torch.conv2')

   local xx = torch.Tensor(2,x:size(1),x:size(2))
   xx[1]:copy(x)
   xx[2]:copy(x)
   local kk = torch.Tensor(2,k:size(1),k:size(2))
   kk[1]:copy(k)
   kk[2]:copy(k)

   local immvc = torch.conv2(xx,kk)
   local immvc2 = torch.conv2(xx,kk,'V')
   local immfc = torch.conv2(xx,kk,'F')

   mytester:asserteq(maxdiff(immvc[1],immvc[2]),0,'torch.conv2')
   mytester:asserteq(maxdiff(immvc[1],imvc),0,'torch.conv2')
   mytester:asserteq(maxdiff(immvc2[1],imvc2),0,'torch.conv2')
   mytester:asserteq(maxdiff(immfc[1],immfc[2]),0,'torch.conv2')
   mytester:asserteq(maxdiff(immfc[1],imfc),0,'torch.conv2')
end

function torchtest.conv3()
   local x = torch.rand(math.floor(torch.uniform(20,40)),
                        math.floor(torch.uniform(20,40)),
                        math.floor(torch.uniform(20,40)))
   local k = torch.rand(math.floor(torch.uniform(5,10)),
                        math.floor(torch.uniform(5,10)),
                        math.floor(torch.uniform(5,10)))
   local imvc = torch.conv3(x,k)
   local imvc2 = torch.conv3(x,k,'V')
   local imfc = torch.conv3(x,k,'F')

   local ki = k:clone();
   local ks = k:storage()
   local kis = ki:storage()
   for i=ks:size(),1,-1 do kis[ks:size()-i+1]=ks[i] end
   local imvx = torch.xcorr3(x,ki)
   local imvx2 = torch.xcorr3(x,ki,'V')
   local imfx = torch.xcorr3(x,ki,'F')

   mytester:asserteq(maxdiff(imvc,imvc2),0,'torch.conv3')
   mytester:asserteq(maxdiff(imvc,imvx),0,'torch.conv3')
   mytester:asserteq(maxdiff(imvc,imvx2),0,'torch.conv3')
   mytester:asserteq(maxdiff(imfc,imfx),0,'torch.conv3')
   mytester:assertlt(math.abs(x:dot(x)-torch.xcorr3(x,x)[1][1][1]),4*1e-10,'torch.conv3')

   local xx = torch.Tensor(2,x:size(1),x:size(2),x:size(3))
   xx[1]:copy(x)
   xx[2]:copy(x)
   local kk = torch.Tensor(2,k:size(1),k:size(2),k:size(3))
   kk[1]:copy(k)
   kk[2]:copy(k)

   local immvc = torch.conv3(xx,kk)
   local immvc2 = torch.conv3(xx,kk,'V')
   local immfc = torch.conv3(xx,kk,'F')

   mytester:asserteq(maxdiff(immvc[1],immvc[2]),0,'torch.conv3')
   mytester:asserteq(maxdiff(immvc[1],imvc),0,'torch.conv3')
   mytester:asserteq(maxdiff(immvc2[1],imvc2),0,'torch.conv3')
   mytester:asserteq(maxdiff(immfc[1],immfc[2]),0,'torch.conv3')
   mytester:asserteq(maxdiff(immfc[1],imfc),0,'torch.conv3')
end

function torchtest.xcorr3_xcorr2_eq()
    local ix = math.floor(torch.uniform(20,40))
    local iy = math.floor(torch.uniform(20,40))
    local iz = math.floor(torch.uniform(20,40))
    local kx = math.floor(torch.uniform(5,10))
    local ky = math.floor(torch.uniform(5,10))
    local kz = math.floor(torch.uniform(5,10))

    local x = torch.rand(ix,iy,iz)
    local k = torch.rand(kx,ky,kz)

    local o3 = torch.xcorr3(x,k)
    local o32 = torch.zeros(o3:size())

    for i=1,o3:size(1) do
        for j=1,k:size(1) do
            o32[i]:add(torch.xcorr2(x[i+j-1],k[j]))
        end
    end

    mytester:assertlt(maxdiff(o3,o32),precision,'torch.conv3_conv2_eq')
end

function torchtest.fxcorr3_fxcorr2_eq()
    local ix = math.floor(torch.uniform(20,40))
    local iy = math.floor(torch.uniform(20,40))
    local iz = math.floor(torch.uniform(20,40))
    local kx = math.floor(torch.uniform(5,10))
    local ky = math.floor(torch.uniform(5,10))
    local kz = math.floor(torch.uniform(5,10))

    local x = torch.rand(ix,iy,iz)
    local k = torch.rand(kx,ky,kz)

    local o3 = torch.xcorr3(x,k,'F')

    local o32 = torch.zeros(o3:size())

    for i=1,x:size(1) do
        for j=1,k:size(1) do
            o32[i+j-1]:add(torch.xcorr2(x[i],k[k:size(1)-j + 1],'F'))
        end
    end

    mytester:assertlt(maxdiff(o3,o32),precision,'torch.conv3_conv2_eq')
end

function torchtest.conv3_conv2_eq()
    local ix = math.floor(torch.uniform(20,40))
    local iy = math.floor(torch.uniform(20,40))
    local iz = math.floor(torch.uniform(20,40))
    local kx = math.floor(torch.uniform(5,10))
    local ky = math.floor(torch.uniform(5,10))
    local kz = math.floor(torch.uniform(5,10))

    local x = torch.rand(ix,iy,iz)
    local k = torch.rand(kx,ky,kz)

    local o3 = torch.conv3(x,k)
    local o32 = torch.zeros(o3:size())

    for i=1,o3:size(1) do
        for j=1,k:size(1) do
            o32[i]:add(torch.conv2(x[i+j-1],k[k:size(1)-j+1]))
        end
    end

    mytester:assertlt(maxdiff(o3,o32),precision,'torch.conv3_conv2_eq')
end

function torchtest.fconv3_fconv2_eq()
    local ix = math.floor(torch.uniform(20,40))
    local iy = math.floor(torch.uniform(20,40))
    local iz = math.floor(torch.uniform(20,40))
    local kx = math.floor(torch.uniform(5,10))
    local ky = math.floor(torch.uniform(5,10))
    local kz = math.floor(torch.uniform(5,10))

    local x = torch.rand(ix,iy,iz)
    local k = torch.rand(kx,ky,kz)

    local o3 = torch.conv3(x,k,'F')

    local o32 = torch.zeros(o3:size())

    for i=1,x:size(1) do
        for j=1,k:size(1) do
            o32[i+j-1]:add(torch.conv2(x[i],k[j],'F'))
        end
    end

    mytester:assertlt(maxdiff(o3,o32),precision,'torch.conv3_conv2_eq')
end

function torchtest.logical()
   local x = torch.rand(100,100)*2-1;
   local xx = x:clone()

   local xgt = torch.gt(x,1)
   local xlt = torch.lt(x,1)

   local xeq = torch.eq(x,1)
   local xne = torch.ne(x,1)

   local neqs = xgt+xlt
   local all = neqs + xeq
   mytester:asserteq(neqs:sum(), xne:sum(), 'torch.logical')
   mytester:asserteq(x:nElement(),all:double():sum() , 'torch.logical')
end

function torchtest.RNGState()
   local state = torch.getRNGState()
   local stateCloned = state:clone()
   local before = torch.rand(1000)

   mytester:assert(state:ne(stateCloned):long():sum() == 0, 'getRNGState should have value semantics, but appears to have reference semantics')

   torch.setRNGState(state)
   local after = torch.rand(1000)
   mytester:assertTensorEq(before, after, 1e-16, 'getRNGState/setRNGState not generating same sequence')
end

function torchtest.RNGStateAliasing()
    torch.manualSeed(1)
    local unused = torch.uniform()

    -- Fork the random number stream at this point
    local gen = torch.Generator()
    torch.setRNGState(gen, torch.getRNGState())

    local target_value = torch.rand(1000)
    --Dramatically alter the internal state of the main generator
    local also_unused = torch.rand(100000)
    local forked_value = torch.rand(gen, 1000)
    mytester:assertTensorEq(target_value, forked_value, 1e-16, "RNG has not forked correctly.")
end

function torchtest.serializeGenerator()
   local generator = torch.Generator()
   torch.manualSeed(generator, 123)
   local differentGenerator = torch.Generator()
   torch.manualSeed(differentGenerator, 124)
   local serializedGenerator = torch.serialize(generator)
   local deserializedGenerator = torch.deserialize(serializedGenerator)
   local generated = torch.random(generator)
   local differentGenerated = torch.random(differentGenerator)
   local deserializedGenerated = torch.random(deserializedGenerator)
   mytester:asserteq(generated, deserializedGenerated, 'torch.Generator changed internal state after being serialized')
   mytester:assertne(generated, differentGenerated, 'Generators with different random seed should not produce the same output')
end

function torchtest.testBoxMullerState()
    torch.manualSeed(123)
    local odd_number = 101
    local seeded = torch.randn(odd_number)
    local state = torch.getRNGState()
    local midstream = torch.randn(odd_number)
    torch.setRNGState(state)
    local repeat_midstream = torch.randn(odd_number)
    torch.manualSeed(123)
    local reseeded = torch.randn(odd_number)
    mytester:assertTensorEq(midstream, repeat_midstream, 1e-16, 'getRNGState/setRNGState not generating same sequence of normally distributed numbers')
    mytester:assertTensorEq(seeded, reseeded, 1e-16, 'repeated calls to manualSeed not generating same sequence of normally distributed numbers')
end

function torchtest.testCholesky()
   local x = torch.rand(10,10)
   local A = torch.mm(x, x:t())

   ---- Default Case
   local C = torch.potrf(A)
   local B = torch.mm(C:t(), C)
   mytester:assertTensorEq(A, B, 1e-14, 'potrf did not allow rebuilding the original matrix')

    ---- Test Upper Triangular
    local U = torch.potrf(A, 'U')
          B = torch.mm(U:t(), U)
    mytester:assertTensorEq(A, B, 1e-14, 'potrf (upper) did not allow rebuilding the original matrix')

    ---- Test Lower Triangular
    local L = torch.potrf(A, 'L')
          B = torch.mm(L, L:t())
    mytester:assertTensorEq(A, B, 1e-14, 'potrf (lower) did not allow rebuilding the original matrix')
end

function torchtest.potrs()
   if not torch.potrs then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                         {-6.05, -3.30,  5.36, -4.44,  1.08},
                         {-0.45,  2.58, -2.70,  0.27,  9.04},
                         {8.32,  2.71,  4.35, -7.17,  2.14},
                         {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()
   local b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                         {-1.56,  4.00, -8.67,  1.75,  2.86},
                         {9.81, -4.09, -4.57, -8.61,  8.99}}):t()

   ---- Make sure 'a' is symmetric PSD
   a = torch.mm(a, a:t())

   ---- Upper Triangular Test
   local U = torch.potrf(a, 'U')
   local x = torch.potrs(b, U, 'U')
   mytester:assertlt(b:dist(a*x),1e-12,"torch.potrs; uplo='U'")

   ---- Lower Triangular Test
   local L = torch.potrf(a, 'L')
   x = torch.potrs(b, L, 'L')
   mytester:assertlt(b:dist(a*x),1e-12,"torch.potrs; uplo='L")
end

function torchtest.potri()
   if not torch.potrs then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                         {-6.05, -3.30,  5.36, -4.44,  1.08},
                         {-0.45,  2.58, -2.70,  0.27,  9.04},
                         {8.32,  2.71,  4.35, -7.17,  2.14},
                         {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()

   ---- Make sure 'a' is symmetric PSD
   a = torch.mm(a, a:t())

   ---- Compute inverse directly
   local inv0 = torch.inverse(a)

   ---- Default case
   local chol = torch.potrf(a)
   local inv1 = torch.potri(chol)
   mytester:assertlt(inv0:dist(inv1),1e-12,"torch.potri; uplo=''")

   ---- Upper Triangular Test
   chol = torch.potrf(a, 'U')
   inv1 = torch.potri(chol, 'U')
   mytester:assertlt(inv0:dist(inv1),1e-12,"torch.potri; uplo='U'")

   ---- Lower Triangular Test
   chol = torch.potrf(a, 'L')
   inv1 = torch.potri(chol, 'L')
   mytester:assertlt(inv0:dist(inv1),1e-12,"torch.potri; uplo='L'")
end

function torchtest.pstrf()
  local function checkPsdCholesky(a, uplo, inplace)
    local u, piv, args, a_reconstructed
    if inplace then
      u = torch.Tensor(a:size())
      piv = torch.IntTensor(a:size(1))
      args = {u, piv, a}
    else
      args = {a}
    end

    if uplo then table.insert(args, uplo) end

    u, piv = torch.pstrf(unpack(args))

    if uplo == 'L' then
      a_reconstructed = torch.mm(u, u:t())
    else
      a_reconstructed = torch.mm(u:t(), u)
    end

    piv = piv:long()
    local a_permuted = a:index(1, piv):index(2, piv)
    mytester:assertTensorEq(a_permuted, a_reconstructed, 1e-14,
                            'torch.pstrf did not allow rebuilding the original matrix;' ..
                            'uplo=' .. tostring(uplo))
  end

  local dimensions = { {5, 1}, {5, 3}, {5, 5}, {10, 10} }
  for _, dim in pairs(dimensions) do
    local m = torch.Tensor(unpack(dim)):uniform()
    local a = torch.mm(m, m:t())
    -- add a small number to the diagonal to make the matrix numerically positive semidefinite
    for i = 1, m:size(1) do
      a[i][i] = a[i][i] + 1e-7
    end
    checkPsdCholesky(a, nil, false)
    checkPsdCholesky(a, 'U', false)
    checkPsdCholesky(a, 'L', false)
    checkPsdCholesky(a, nil, true)
    checkPsdCholesky(a, 'U', true)
    checkPsdCholesky(a, 'L', true)
  end
end

function torchtest.testNumel()
    local b = torch.ByteTensor(3, 100, 100)
    mytester:asserteq(b:nElement(), 3*100*100, "nElement not right")
    mytester:asserteq(b:numel(), 3*100*100, "numel not right")
end


-- Generate a tensor of size `size` whose values are ascending integers from
-- `start` (or 1, if `start is not given)
local function consecutive(size, start)
    local sequence = torch.ones(torch.Tensor(size):prod(1)[1]):cumsum(1)
    if start then
        sequence:add(start - 1)
    end
    return sequence:resize(unpack(size))
end

function torchtest.index()
    local badIndexMsg = "Lookup with valid index should return correct result"
    local reference = consecutive{3, 3, 3}
    mytester:assertTensorEq(reference[1], consecutive{3, 3}, 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[2], consecutive({3, 3}, 10), 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[3], consecutive({3, 3}, 19), 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[{1}], consecutive{3, 3}, 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[{2}], consecutive({3, 3}, 10), 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[{3}], consecutive({3, 3}, 19), 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[{1,2}], consecutive({3}, 4), 1e-16, badIndexMsg)
    mytester:assertTensorEq(reference[{{1,2}}], consecutive({2, 3, 3}), 1e-16, badIndexMsg)
    mytester:asserteq(reference[{3, 3, 3}], 27, badIndexMsg)
    mytester:assertTensorEq(reference[{}], consecutive{3, 3, 3}, 1e-16, badIndexMsg)

    local shouldErrorMsg = "Lookup with too many indices should error"
    mytester:assertError(function() return reference[{1, 1, 1, 1}] end, shouldErrorMsg)
    mytester:assertError(function() return reference[{1, 1, 1, {1, 1}}] end, shouldErrorMsg)
    mytester:assertError(function() return reference[{3, 3, 3, 3, 3, 3, 3, 3}] end, shouldErrorMsg)
end

function torchtest.newIndex()
    local badIndexMsg = "Assignment to valid index should produce correct result"
    local reference = consecutive{3, 3, 3}
    -- This relies on __index__() being correct - but we have separate tests for that
    local function checkPartialAssign(index)
        local reference = torch.zeros(3, 3, 3)
        reference[index] = consecutive{3, 3, 3}[index]
        mytester:assertTensorEq(reference[index], consecutive{3, 3, 3}[index], 1e-16, badIndexMsg)
        reference[index] = 0
        mytester:assertTensorEq(reference, torch.zeros(3, 3, 3), 1e-16, badIndexMsg)
    end

    checkPartialAssign{1}
    checkPartialAssign{2}
    checkPartialAssign{3}
    checkPartialAssign{1,2}
    checkPartialAssign{2,3}
    checkPartialAssign{1,3}
    checkPartialAssign{}

    local shouldErrorMsg = "Assignment with too many indices should error"
    mytester:assertError(function() reference[{1, 1, 1, 1}] = 1 end, shouldErrorMsg)
    mytester:assertError(function() reference[{1, 1, 1, {1, 1}}] = 1 end, shouldErrorMsg)
    mytester:assertError(function() reference[{3, 3, 3, 3, 3, 3, 3, 3}] = 1 end, shouldErrorMsg)
end

function torchtest.indexCopy()
   local nCopy, nDest = 3, 20
   local dest = torch.randn(nDest,4,5)
   local src = torch.randn(nCopy,4,5)
   local idx = torch.randperm(nDest):narrow(1, 1, nCopy):long()
   local dest2 = dest:clone()
   dest:indexCopy(1, idx, src)
   for i=1,idx:size(1) do
      dest2[idx[i]]:copy(src[i])
   end
   mytester:assertTensorEq(dest, dest2, 0.000001, "indexCopy tensor error")

   local dest = torch.randn(nDest)
   local src = torch.randn(nCopy)
   local idx = torch.randperm(nDest):narrow(1, 1, nCopy):long()
   local dest2 = dest:clone()
   dest:indexCopy(1, idx, src)
   for i=1,idx:size(1) do
      dest2[idx[i]] = src[i]
   end
   mytester:assertTensorEq(dest, dest2, 0.000001, "indexCopy scalar error")
end

function torchtest.indexAdd()
   local nCopy, nDest = 3, 20
   local dest = torch.randn(nDest,4,5)
   local src = torch.randn(nCopy,4,5)
   local idx = torch.randperm(nDest):narrow(1, 1, nCopy):long()
   local dest2 = dest:clone()
   dest:indexAdd(1, idx, src)
   for i=1,idx:size(1) do
      dest2[idx[i]]:add(src[i])
   end
   mytester:assertTensorEq(dest, dest2, 0.000001, "indexAdd tensor error")

   local dest = torch.randn(nDest)
   local src = torch.randn(nCopy)
   local idx = torch.randperm(nDest):narrow(1, 1, nCopy):long()
   local dest2 = dest:clone()
   dest:indexAdd(1, idx, src)
   for i=1,idx:size(1) do
      dest2[idx[i]] = dest2[idx[i]] + src[i]
   end
   mytester:assertTensorEq(dest, dest2, 0.000001, "indexAdd scalar error")
end

-- Fill idx with valid indices.
local function fillIdx(idx, dim, dim_size, elems_per_row, m, n, o)
   for i = 1, (dim == 1 and 1 or m) do
      for j = 1, (dim == 2 and 1 or n) do
         for k = 1, (dim == 3 and 1 or o) do
            local ii = {i, j, k}
            ii[dim] = {}
            idx[ii] = torch.randperm(dim_size)[{{1, elems_per_row}}]
         end
      end
   end
end

function torchtest.gather()
   local m, n, o = torch.random(10, 20), torch.random(10, 20), torch.random(10, 20)
   local elems_per_row = torch.random(10)
   local dim = torch.random(3)

   local src = torch.randn(m, n, o)
   local idx_size = {m, n, o}
   idx_size[dim] = elems_per_row
   local idx = torch.LongTensor():resize(unpack(idx_size))
   fillIdx(idx, dim, src:size(dim), elems_per_row, m, n, o)

   local actual = torch.gather(src, dim, idx)
   local expected = torch.Tensor():resize(unpack(idx_size))
   for i = 1, idx_size[1] do
      for j = 1, idx_size[2] do
         for k = 1, idx_size[3] do
            local ii = {i, j, k}
            ii[dim] = idx[i][j][k]
            expected[i][j][k] = src[ii]
         end
      end
   end
   mytester:assertTensorEq(actual, expected, 0, "Wrong values for gather")

   idx[1][1][1] = 23
   mytester:assertError(function() torch.gather(src, dim, idx) end,
                        "Invalid index not detected")
end

function torchtest.gatherMax()
   local src = torch.randn(3, 4, 5)
   local expected, idx = src:max(3)
   local actual = torch.gather(src, 3, idx)
   mytester:assertTensorEq(actual, expected, 0, "Wrong values for gather")
end

function torchtest.scatter()
   local m, n, o = torch.random(10, 20), torch.random(10, 20), torch.random(10, 20)
   local elems_per_row = torch.random(10)
   local dim = torch.random(3)

   local idx_size = {m, n, o}
   idx_size[dim] = elems_per_row
   local idx = torch.LongTensor():resize(unpack(idx_size))
   fillIdx(idx, dim, ({m, n, o})[dim], elems_per_row, m, n, o)
   local src = torch.Tensor():resize(unpack(idx_size)):normal()

   local actual = torch.zeros(m, n, o):scatter(dim, idx, src)
   local expected = torch.zeros(m, n, o)
   for i = 1, idx_size[1] do
      for j = 1, idx_size[2] do
         for k = 1, idx_size[3] do
            local ii = {i, j, k}
            ii[dim] = idx[i][j][k]
           expected[ii] = src[i][j][k]
         end
      end
   end
   mytester:assertTensorEq(actual, expected, 0, "Wrong values for scatter")

   idx[1][1][1] = 34
   mytester:assertError(function() torch.zeros(m, n, o):scatter(dim, idx, src) end,
                        "Invalid index not detected")
end

function torchtest.scatterFill()
   local m, n, o = torch.random(10, 20), torch.random(10, 20), torch.random(10, 20)
   local elems_per_row = torch.random(10)
   local dim = torch.random(3)

   local val = torch.uniform()
   local idx_size = {m, n, o}
   idx_size[dim] = elems_per_row
   local idx = torch.LongTensor():resize(unpack(idx_size))
   fillIdx(idx, dim, ({m, n, o})[dim], elems_per_row, m, n, o)

   local actual = torch.zeros(m, n, o):scatter(dim, idx, val)
   local expected = torch.zeros(m, n, o)
   for i = 1, idx_size[1] do
      for j = 1, idx_size[2] do
         for k = 1, idx_size[3] do
            local ii = {i, j, k}
            ii[dim] = idx[i][j][k]
            expected[ii] = val
         end
      end
   end
   mytester:assertTensorEq(actual, expected, 0, "Wrong values for scatter")

   idx[1][1][1] = 28
   mytester:assertError(function() torch.zeros(m, n, o):scatter(dim, idx, val) end,
                        "Invalid index not detected")
end

function torchtest.maskedCopy()
   local nCopy, nDest = 3, 10
   local dest = torch.randn(nDest)
   local src = torch.randn(nCopy)
   local mask = torch.ByteTensor{0,0,0,0,1,0,1,0,1,0}
   local dest2 = dest:clone()
   dest:maskedCopy(mask, src)
   local j = 1
   for i=1,nDest do
      if mask[i] == 1 then
         dest2[i] = src[j]
         j = j + 1
      end
   end
   mytester:assertTensorEq(dest, dest2, 0.000001, "maskedCopy error")

   -- make source bigger than number of 1s in mask
   src = torch.randn(nDest)
   local ok = pcall(dest.maskedCopy, dest, mask, src)
   mytester:assert(ok, "maskedCopy incorrect complaint when"
		      .. " src is bigger than mask's one count")

   src = torch.randn(nCopy - 1) -- make src smaller. this should fail
   local ok = pcall(dest.maskedCopy, dest, mask, src)
   mytester:assert(not ok, "maskedCopy not erroring when"
		      .. " src is smaller than mask's one count")
end

function torchtest.maskedSelect()
   local nSrc = 10
   local src = torch.randn(nSrc)
   local mask = torch.rand(nSrc):mul(2):floor():byte()
   local dst = torch.Tensor()
   dst:maskedSelect(src, mask)
   local dst2 = {}
   for i=1,nSrc do
      if mask[i] == 1 then
         table.insert(dst2, src[i])
      end
   end
   mytester:assertTensorEq(dst, torch.DoubleTensor(dst2), 0.000001, "maskedSelect error")
end

function torchtest.maskedFill()
   local nDst = 10
   local dst = torch.randn(nDst)
   local mask = torch.rand(nDst):mul(2):floor():byte()
   local val = math.random()
   local dst2 = dst:clone()
   dst:maskedFill(mask, val)
   for i=1,nDst do
      if mask[i] == 1 then
         dst2[i] = val
      end
   end
   mytester:assertTensorEq(dst, dst2, 0.000001, "maskedFill error")
end

function torchtest.abs()
   local size = 1000
   local range = 1000
   local original = torch.rand(size):mul(range)
   -- Tensor filled with {-1,1}
   local switch = torch.rand(size):mul(2):floor():mul(2):add(-1)

   local types = {'torch.DoubleTensor', 'torch.FloatTensor', 'torch.LongTensor', 'torch.IntTensor'}
   for k,t in ipairs(types) do
      local data = original:type(t)
      local switch = switch:type(t)
      local input = torch.cmul(data, switch)
      mytester:assertTensorEq(input:abs(), data, 1e-16, 'Error in abs() for '..t)
   end

   -- Checking that the right abs function is called for LongTensor
   local bignumber
   if torch.LongTensor():elementSize() > 4 then
      bignumber = 2^31 + 1
   else
      bignumber = 2^15 + 1
   end
   local input = torch.LongTensor{-bignumber}
   mytester:assertgt(input:abs()[1], 0, 'torch.abs(3)')
end

function torchtest.classInModule()
    -- Need a global for this module
    _mymodule123 = {}
    local x = torch.class('_mymodule123.myclass')
    mytester:assert(x ~= nil, 'Could not create class in module')
    -- Remove the global
    _G['_mymodule123'] = nil
    debug.getregistry()['_mymodule123.myclass']=nil
end

function torchtest.classNoModule()
    local x = torch.class('_myclass123')
    mytester:assert(x ~= nil, 'Could not create class in module')
    debug.getregistry()['_myclass123'] = nil
end

function torchtest.type()
   local objects = {torch.DoubleTensor(), {}, nil, 2, "asdf"}
   local types = {'torch.DoubleTensor', 'table', 'nil', 'number', 'string'}
   for i,obj in ipairs(objects) do
      mytester:assert(torch.type(obj) == types[i], "wrong type "..types[i])
   end
end

function torchtest.isTypeOfInheritance()
   do
      local A = torch.class('A')
      local B, parB = torch.class('B', 'A')
      local C, parC = torch.class('C', 'A')
   end
   local a, b, c = A(), B(), C()

   mytester:assert(torch.isTypeOf(a, 'A'), 'isTypeOf error, string spec')
   mytester:assert(torch.isTypeOf(a, A), 'isTypeOf error, constructor')
   mytester:assert(torch.isTypeOf(b, 'B'), 'isTypeOf error child class')
   mytester:assert(torch.isTypeOf(b, B), 'isTypeOf error child class ctor')
   mytester:assert(torch.isTypeOf(b, 'A'), 'isTypeOf error: inheritance')
   mytester:assert(torch.isTypeOf(b, A), 'isTypeOf error: inheritance')
   mytester:assert(not torch.isTypeOf(c, 'B'), 'isTypeOf error: common parent')
   mytester:assert(not torch.isTypeOf(c, B), 'isTypeOf error: common parent')
   debug.getregistry()['A'] = nil
   debug.getregistry()['B'] = nil
   debug.getregistry()['C'] = nil
end

function torchtest.isTypeOfPartial()
    do
      local TorchDummy = torch.class('TorchDummy')
      local OtherTorchDummy = torch.class('OtherTorchDummy')
      local TorchMember = torch.class('TorchMember')
      local OtherTorchMember = torch.class('OtherTorchMember')
      local FirstTorchMember = torch.class('FirstTorchMember',
                                           'TorchMember')
      local SecondTorchMember = torch.class('SecondTorchMember',
                                            'TorchMember')
      local ThirdTorchMember = torch.class('ThirdTorchMember',
                                           'OtherTorchMember')
   end
   local td, otd = TorchDummy(), OtherTorchDummy()
   local tm, ftm, stm, ttm = TorchMember(), FirstTorchMember(),
      SecondTorchMember(), ThirdTorchMember()

   mytester:assert(not torch.isTypeOf(td, 'OtherTorchDummy'),
                   'isTypeOf error: incorrect partial match')
   mytester:assert(not torch.isTypeOf(otd, 'TorchDummy'),
                   'isTypeOf error: incorrect partial match')
   mytester:assert(torch.isTypeOf(tm, 'TorchMember'),
                   'isTypeOf error, string spec')
   mytester:assert(torch.isTypeOf(tm, TorchMember),
                   'isTypeOf error, constructor')
   mytester:assert(torch.isTypeOf(ftm, 'FirstTorchMember'),
                   'isTypeOf error child class')
   mytester:assert(torch.isTypeOf(ftm, FirstTorchMember),
                   'isTypeOf error child class ctor')
   mytester:assert(torch.isTypeOf(ftm, 'TorchMember'),
                   'isTypeOf error: inheritance')
   mytester:assert(torch.isTypeOf(ftm, TorchMember),
                   'isTypeOf error: inheritance')
   mytester:assert(not torch.isTypeOf(stm, 'FirstTorchMember'),
                   'isTypeOf error: common parent')
   mytester:assert(not torch.isTypeOf(stm, FirstTorchMember),
                   'isTypeOf error: common parent')
   mytester:assert(not torch.isTypeOf(ttm, TorchMember),
                   'isTypeOf error: inheritance')
   mytester:assert(not torch.isTypeOf(ttm, 'TorchMember'),
                   'isTypeOf error: inheritance')
   debug.getregistry()['TorchDummy'] = nil
   debug.getregistry()['OtherTorchDummy'] = nil
   debug.getregistry()['TorchMember'] = nil
   debug.getregistry()['OtherTorchMember'] = nil
   debug.getregistry()['FirstTorchMember'] = nil
   debug.getregistry()['SecondTorchMember'] = nil
   debug.getregistry()['ThirdTorchMember'] = nil
end

function torchtest.isTypeOfPattern()
   local t = torch.LongTensor()
   mytester:assert(torch.isTypeOf(t, torch.LongTensor),
                   'isTypeOf error: incorrect match')
   mytester:assert(not torch.isTypeOf(t, torch.IntTensor),
                   'isTypeOf error: incorrect match')
   mytester:assert(torch.isTypeOf(t, 'torch.LongTensor'),
                   'isTypeOf error: incorrect match')
   mytester:assert(not torch.isTypeOf(t, 'torch.Long'),
                   'isTypeOf error: incorrect match')
   mytester:assert(torch.isTypeOf(t, 'torch.*Tensor'),
                   'isTypeOf error: incorrect match')
   mytester:assert(torch.isTypeOf(t, '.*Long'),
                   'isTypeOf error: incorrect match')
   mytester:assert(not torch.isTypeOf(t, 'torch.IntTensor'),
                   'isTypeOf error: incorrect match')
end

function torchtest.isTensor()
   for k,v in ipairs({"real", "half"}) do
      torchtest_isTensor(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_isTensor(func)
   local t = func(torch.randn(3,4))
   mytester:assert(torch.isTensor(t), 'error in isTensor')
   mytester:assert(torch.isTensor(t[1]), 'error in isTensor for subTensor')
   mytester:assert(not torch.isTensor(t[1][2]), 'false positive in isTensor')
   mytester:assert(torch.Tensor.isTensor(t), 'alias not working')
end

function torchtest.isStorage()
   for k,v in ipairs({"real", "half"}) do
      torchtest_isStorage(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_isStorage(func)
  local t = torch.randn(3,4)
  mytester:assert(torch.isStorage(t:storage()), 'error in isStorage')
  mytester:assert(not torch.isStorage(t), 'false positive in isStorage')
end

function torchtest.view()
   for k,v in ipairs({"real", "half"}) do
      torchtest_view(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_view(func)
   local tensor = func(torch.rand(15))
   local template = func(torch.rand(3,5))
   local target = template:size():totable()
   mytester:assertTableEq(tensor:viewAs(template):size():totable(), target, 'Error in viewAs')
   mytester:assertTableEq(tensor:view(3,5):size():totable(), target, 'Error in view')
   mytester:assertTableEq(tensor:view(torch.LongStorage{3,5}):size():totable(), target, 'Error in view using LongStorage')
   mytester:assertTableEq(tensor:view(-1,5):size():totable(), target, 'Error in view using dimension -1')
   mytester:assertTableEq(tensor:view(3,-1):size():totable(), target, 'Error in view using dimension -1')
   local tensor_view = tensor:view(5,3)
   tensor_view:fill(torch.rand(1)[1])
   mytester:asserteq((tensor_view-tensor):abs():max(), 0, 'Error in view')

   local target_tensor = func(torch.Tensor())
   mytester:assertTableEq(target_tensor:viewAs(tensor, template):size():totable(), target, 'Error in viewAs')
   mytester:assertTableEq(target_tensor:view(tensor, 3,5):size():totable(), target, 'Error in view')
   mytester:assertTableEq(target_tensor:view(tensor, torch.LongStorage{3,5}):size():totable(), target, 'Error in view using LongStorage')
   mytester:assertTableEq(target_tensor:view(tensor, -1,5):size():totable(), target, 'Error in view using dimension -1')
   mytester:assertTableEq(target_tensor:view(tensor, 3,-1):size():totable(), target, 'Error in view using dimension -1')
   target_tensor:fill(torch.rand(1)[1])
   mytester:asserteq((target_tensor-tensor):abs():max(), 0, 'Error in viewAs')
end

function torchtest.expand()
   for k,v in ipairs({"real", "half"}) do
      torchtest_expand(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_expand(func)
   local result = func(torch.Tensor())
   local tensor = func(torch.rand(8,1))
   local template = func(torch.rand(8,5))
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
   for k,v in ipairs({"real", "half"}) do
      torchtest_repeatTensor(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_repeatTensor(func, mean)
   local result = func(torch.Tensor())
   local tensor = func(torch.rand(8,4))
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
   for k,v in ipairs({"real", "half"}) do
      torchtest_isSameSizeAs(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_isSameSizeAs(func)
   local t1 = func(torch.Tensor(3, 4, 9, 10))
   local t2 = func(torch.Tensor(3, 4))
   local t3 = func(torch.Tensor(1, 9, 3, 3))
   local t4 = func(torch.Tensor(3, 4, 9, 10))

   mytester:assert(t1:isSameSizeAs(t2) == false, "wrong answer ")
   mytester:assert(t1:isSameSizeAs(t3) == false, "wrong answer ")
   mytester:assert(t1:isSameSizeAs(t4) == true, "wrong answer ")
end

function torchtest.isSetTo()
   for k,v in ipairs({"real", "half"}) do
      torchtest_isSetTo(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_isSetTo(func)
   local t1 = func(torch.Tensor(3, 4, 9, 10))
   local t2 = func(torch.Tensor(3, 4, 9, 10))
   local t3 = func(torch.Tensor()):set(t1)
   local t4 = t3:reshape(12, 90)
   mytester:assert(t1:isSetTo(t2) == false, "tensors do not share storage")
   mytester:assert(t1:isSetTo(t3) == true, "tensor is set to other")
   mytester:assert(t3:isSetTo(t1) == true, "isSetTo should be symmetric")
   mytester:assert(t1:isSetTo(t4) == false, "tensors have different view")
   mytester:assert(not func(torch.Tensor()):isSetTo(func(torch.Tensor())),
                   "Tensors with no storages should not appear to be set " ..
                   "to each other")
end

function torchtest.equal()
  -- Contiguous, 1D
  local t1 = torch.Tensor{3, 4, 9, 10}
  local t2 = t1:clone()
  local t3 = torch.Tensor{1, 9, 3, 10}
  local t4 = torch.Tensor{3, 4, 9}
  local t5 = torch.Tensor()
  mytester:assert(t1:equal(t2) == true, "wrong answer ")
  mytester:assert(t1:equal(t3) == false, "wrong answer ")
  mytester:assert(t1:equal(t4) == false, "wrong answer ")
  mytester:assert(t1:equal(t5) == false, "wrong answer ")
  mytester:assert(torch.equal(t1, t2) == true, "wrong answer ")
  mytester:assert(torch.equal(t1, t3) == false, "wrong answer ")
  mytester:assert(torch.equal(t1, t4) == false, "wrong answer ")
  mytester:assert(torch.equal(t1, t5) == false, "wrong answer ")

  -- Non contiguous, 2D
  local s = torch.Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}})
  local s1 = s[{{}, {2, 3}}]
  local s2 = s1:clone()
  local s3 = torch.Tensor({{2, 3}, {6, 7}})
  local s4 = torch.Tensor({{0, 0}, {0, 0}})

  mytester:assert(not s1:isContiguous(), "wrong answer ")
  mytester:assert(s1:equal(s2) == true, "wrong answer ")
  mytester:assert(s1:equal(s3) == true, "wrong answer ")
  mytester:assert(s1:equal(s4) == false, "wrong answer ")
  mytester:assert(torch.equal(s1, s2) == true, "wrong answer ")
  mytester:assert(torch.equal(s1, s3) == true, "wrong answer ")
  mytester:assert(torch.equal(s1, s4) == false, "wrong answer ")
end

function torchtest.isSize()
   for k,v in ipairs({"real", "half"}) do
      torchtest_isSize(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_isSize(func)
  local t1 = func(torch.Tensor(3, 4, 5))
  local s1 = torch.LongStorage({3, 4, 5})
  local s2 = torch.LongStorage({5, 4, 3})

   mytester:assert(t1:isSize(s1) == true, "wrong answer ")
   mytester:assert(t1:isSize(s2) == false, "wrong answer ")
   mytester:assert(t1:isSize(t1:size()) == true, "wrong answer ")
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

function torchtest.split()
   for k,v in ipairs({"real", "half"}) do
      torchtest_split(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_split(func)
   local result = {}
   local tensor = func(torch.rand(7,4))
   local splitSize = 3
   local targetSize = {{3,4},{3,4},{1,4}}
   local dim = 1
   local splits = tensor:split(splitSize, dim)
   local start = 1
   for i, split in ipairs(splits) do
      mytester:assertTableEq(split:size():totable(), targetSize[i], 'Size error in split '..i)
      mytester:assertTensorEq(tensor:narrow(dim, start, targetSize[i][dim]), split, 0.00001, 'Content error in split '..i)
      start = start + targetSize[i][dim]
   end
   torch.split(result, tensor, splitSize, dim)
   local start = 1
   for i, split in ipairs(result) do
      mytester:assertTableEq(split:size():totable(), targetSize[i], 'Result size error in split '..i)
      mytester:assertTensorEq(tensor:narrow(dim, start, targetSize[i][dim]), split, 0.000001, 'Result content error in split '..i)
      start = start + targetSize[i][dim]
   end
   mytester:asserteq(#splits, #result, 'Non-consistent output size from split')
   for i, split in ipairs(splits) do
      mytester:assertTensorEq(split,result[i], 0, 'Non-consistent outputs from split')
   end
end

function torchtest.chunk()
   for k,v in ipairs({"real", "half"}) do
      torchtest_chunk(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_chunk(func)
   local result = {}
   local tensor = func(torch.rand(4,7))
   local nChunk = 3
   local targetSize = {{4,3},{4,3},{4,1}}
   local dim = 2
   local splits = tensor:chunk(nChunk, dim)
   local start = 1
   for i, split in ipairs(splits) do
      mytester:assertTableEq(split:size():totable(), targetSize[i], 'Size error in chunk '..i)
      mytester:assertTensorEq(tensor:narrow(dim, start, targetSize[i][dim]), split, 0.00001, 'Content error in chunk '..i)
      start = start + targetSize[i][dim]
   end
   torch.split(result, tensor, nChunk, dim)
   local start = 1
   for i, split in ipairs(result) do
      mytester:assertTableEq(split:size():totable(), targetSize[i], 'Result size error in chunk '..i)
      mytester:assertTensorEq(tensor:narrow(dim, start, targetSize[i][dim]), split, 0.000001, 'Result content error in chunk '..i)
      start = start + targetSize[i][dim]
   end
end

function torchtest.table()
   local convStorage = {
     ['real'] = 'FloatStorage',
     ['half'] = 'HalfStorage'
   }
   for k,v in ipairs(convStorage) do
      torchtest_totable(torch.getmetatable(torch.Tensor():type())[k], v)
   end
end

function torchtest_totable(func, storageType)
  local table0D = {}
  local tensor0D = func(torch.Tensor(table0D))
  mytester:assertTableEq(torch.totable(tensor0D), table0D, 'tensor0D:totable incorrect')

  local table1D = {1, 2, 3}
  local tensor1D = func(torch.Tensor(table1D))
  local storage = torch[storageType](table1D)
  mytester:assertTableEq(tensor1D:totable(), table1D, 'tensor1D:totable incorrect')
  mytester:assertTableEq(storage:totable(), table1D, 'storage:totable incorrect')
  mytester:assertTableEq(torch.totable(tensor1D), table1D, 'torch.totable incorrect for Tensors')
  mytester:assertTableEq(torch.totable(storage), table1D, 'torch.totable incorrect for Storages')

  local table2D = {{1, 2}, {3, 4}}
  local tensor2D = func(torch.Tensor(table2D))
  mytester:assertTableEq(tensor2D:totable(), table2D, 'tensor2D:totable incorrect')

  local tensor3D = func(torch.Tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}))
  local tensorNonContig = tensor3D:select(2, 2)
  mytester:assert(not tensorNonContig:isContiguous(), 'invalid test')
  mytester:assertTableEq(tensorNonContig:totable(), {{3, 4}, {7, 8}},
                         'totable() incorrect for non-contiguous tensors')
end

function torchtest.permute()
   for k,v in ipairs({"real", "half"}) do
      torchtest_permute(torch.getmetatable(torch.Tensor():type())[v])
   end
end

function torchtest_permute(func)
  local orig = {1,2,3,4,5,6,7}
  local perm = torch.randperm(7):totable()
  local x = torch.Tensor(unpack(orig)):fill(0)
  local new = x:permute(unpack(perm)):size():totable()
  mytester:assertTableEq(perm, new, 'Tensor:permute incorrect')
  mytester:assertTableEq(x:size():totable(), orig, 'Tensor:permute changes tensor')
end

function torchtest.serialize()
   local tableObj = {6, a = 42}
   local tensObj = torch.randn(3,4,5)

   -- Test serializing a table
   local serString = torch.serialize(tableObj)
   local serStorage = torch.serializeToStorage(tableObj)
   mytester:assertTableEq(tableObj, torch.deserialize(serString))
   mytester:assertTableEq(tableObj, torch.deserializeFromStorage(serStorage))

   -- Test serializing a Tensor
   serString = torch.serialize(tensObj)
   serStorage = torch.serializeToStorage(tensObj)
   mytester:assertTensorEq(tensObj, torch.deserialize(serString), 1e-10)
   mytester:assertTensorEq(tensObj, torch.deserializeFromStorage(serStorage), 1e-10)
end

function torchtest.storageview()
   local s1 = torch.LongStorage({3, 4, 5})
   local s2 = torch.LongStorage(s1, 2)

   mytester:assert(s2:size() == 2, "should be size 2")
   mytester:assert(s2[1] == s1[2], "should have 4 at position 1")
   mytester:assert(s2[2] == s1[3], "should have 5 at position 2")

   s2[1] = 13
   mytester:assert(13 == s1[2], "should have 13 at position 1")
end

function torchtest.nonzero()
  local nSrc = 12

  local types = {
      'torch.ByteTensor',
      'torch.CharTensor',
      'torch.ShortTensor',
      'torch.IntTensor',
      'torch.FloatTensor',
      'torch.DoubleTensor',
      'torch.LongTensor',
  }

  local shapes = {
      torch.LongStorage{12},
      torch.LongStorage{12, 1},
      torch.LongStorage{1, 12},
      torch.LongStorage{6, 2},
      torch.LongStorage{3, 2, 2},
  }

  for _, type in ipairs(types) do
    local tensor = torch.rand(nSrc):mul(2):floor():type(type)
      for _, shape in ipairs(shapes) do
        tensor = tensor:reshape(shape)
        local dst1 = torch.nonzero(tensor)
        local dst2 = tensor:nonzero()
        -- Does not work. Torch uses the first argument to determine what
        -- type the Tensor is expected to be. In our case the second argument
        -- determines the type of Tensor.
        --local dst3 = torch.LongTensor()
        --torch.nonzero(dst3, tensor)
        -- However, there are workarounds to this issue when it is desired to
        -- use an existing tensor for the result:
        local dst4 = torch.LongTensor()
        tensor.nonzero(dst4, tensor)
        if shape:size() == 1 then
          local dst = {}
          for i = 1 , nSrc do
            if tensor[i] ~= 0 then
              table.insert(dst, i)
            end
          end
          mytester:assertTensorEq(dst1:select(2, 1), torch.LongTensor(dst), 0.0,
                                  "nonzero error")
          mytester:assertTensorEq(dst2:select(2, 1), torch.LongTensor(dst), 0.0,
                                  "nonzero error")
          --mytester:assertTensorEq(dst3:select(2, 1), torch.LongTensor(dst),
          --                        0.0,  "nonzero error")
          mytester:assertTensorEq(dst4:select(2, 1), torch.LongTensor(dst), 0.0,
                                  "nonzero error")
        elseif shape:size() == 2 then
          -- This test will allow through some false positives. It only checks
          -- that the elements flagged positive are indeed non-zero.
          for i=1,dst1:size()[1] do
            mytester:assert(tensor[dst1[i][1]][dst1[i][2]] ~= 0)
          end
        elseif shape:size() == 3 then
          -- This test will allow through some false positives. It only checks
          -- that the elements flagged positive are indeed non-zero.
          for i=1,dst1:size()[1] do
            mytester:assert(tensor[dst1[i][1]][dst1[i][2]][dst1[i][3]] ~= 0)
          end
        end
      end
   end

end

function torchtest.testheaptracking()
  local oldheaptracking = torch._heaptracking
  if oldheaptracking == nil then
    oldheaptracking = false
  end
  torch.setheaptracking(true)
  mytester:assert(torch._heaptracking == true, 'Heap tracking expected true')

  torch.setheaptracking(false)
  mytester:assert(torch._heaptracking == false, 'Heap tracking expected false')

  -- put heap tracking to its original state
  torch.setheaptracking(oldheaptracking)
end

function torchtest.bernoulli()
  local size = torch.LongStorage{10, 10}
  local t = torch.ByteTensor(size)

  local function isBinary(t)
    return torch.ne(t, 0):cmul(torch.ne(t, 1)):sum() == 0
  end

  local p = 0.5
  t:bernoulli(p)
  mytester:assert(isBinary(t), 'Sample from torch.bernoulli is not binary')

  local p = torch.rand(size)
  t:bernoulli(p)
  mytester:assert(isBinary(t), 'Sample from torch.bernoulli is not binary')
end

function torchtest.logNormal()
    local t = torch.FloatTensor(10, 10)
    local mean, std = torch.uniform(), 0.1 * torch.uniform()
    local tolerance = 0.02

    t:logNormal(mean, std)
    local logt = t:log()
    mytester:assertalmosteq(logt:mean(), mean, tolerance, 'mean is wrong')
    mytester:assertalmosteq(logt:std(), std, tolerance, 'tolerance is wrong')
end

function torch.test(tests)
   torch.setheaptracking(true)
   math.randomseed(os.time())
   if torch.getdefaulttensortype() == 'torch.FloatTensor' then
      precision = 1e-4
   elseif  torch.getdefaulttensortype() == 'torch.DoubleTensor' then
      precision = 1e-8
   end
   mytester = torch.Tester()
   mytester:add(torchtest)
   mytester:run(tests)
   return mytester
end
