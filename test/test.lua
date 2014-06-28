--require 'torch'

local mytester 
local torchtest = {}
local msize = 100

local function maxdiff(x,y)
   local d = x-y
   if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.FloatTensor' then
      return d:abs():max()
   else
      local dd = torch.Tensor():resize(d:size()):copy(d)
      return dd:abs():max()
   end
end

function torchtest.dot()
   local v1 = torch.randn(100)
   local v2 = torch.randn(100)

   local res1 = torch.dot(v1,v2)

   local res2 = 0
   for i = 1,v1:size(1) do
      res2 = res2 + v1[i] * v2[i]
   end

   local err = math.abs(res1-res2)
   
   mytester:assertlt(err, precision, 'error in torch.dot')
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
]]

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

function torchtest.max()  -- torch.max([resval, resind,] x [,dim])
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
   local m1 = torch.randn(100,100)
   local res1val, res1ind = torch.max(m1, 2)
   local res2val = res1val:clone():zero()
   local res2ind = res1ind:clone():zero()
   for i=1, m1:size(1) do
      res2val[i] = m1[i][1]
      res2ind[i] = 1
      for j=1, m1:size(2) do
	 if m1[i][j] > res2val[i][1] then
	    res2val[i] = m1[i][j]
	    res2ind[i] = j
	 end
      end
   end
   local errval = res1val:clone():zero()
   for i = 1, res1val:size(1) do
      errval[i] = math.abs(res1val[i][1] - res2val[i][1])
      mytester:asserteq(res1ind[i][1], res2ind[i][1], 'error in torch.max - non-contiguous')
   end
   local maxerr = 0
   for i = 1, errval:size(1) do
      if errval[i][1] > maxerr then
	 maxerr = errval[i]
      end
   end
   mytester:assertlt(maxerr, precision, 'error in torch.max - non-contiguous')      
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
   -- torch.min([resval, resind,] x ,dim])
   local m1 = torch.randn(100,100)
   local res1val, res1ind = torch.min(m1, 2)
   local res2val = res1val:clone():zero()
   local res2ind = res1ind:clone():zero()
   for i=1, m1:size(1) do
      res2val[i] = m1[i][1]
      res2ind[i] = 1
      for j=1, m1:size(2) do
	 if m1[i][j] < res2val[i][1] then
	    res2val[i] = m1[i][j]
	    res2ind[i] = j
	 end
      end
   end
   local errval = res1val:clone():zero()
   for i = 1, res1val:size(1) do
      errval[i] = math.abs(res1val[i][1] - res2val[i][1])
      mytester:asserteq(res1ind[i][1], res2ind[i][1], 'error in torch.min - non-contiguous')
   end
   local minerr = 0
   for i = 1, errval:size(1) do
      if errval[i][1] < minerr then
	 minerr = errval[i]
      end
   end
   mytester:assertlt(minerr, precision, 'error in torch.min - non-contiguous')      
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

function torchtest.add()
   -- [res] torch.add([res,] tensor1, tensor2)
   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.add(m1[{ 4,{} }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(2) do
      res2[i] = m1[4][i] + v1[i]
   end

   local err = (res1-res2):abs():max()
   
   mytester:assertlt(err, precision, 'error in torch.add - contiguous')

   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.add(m1[{ {},4 }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(1) do
      res2[i] = m1[i][4] + v1[i]
   end

   local err = (res1-res2):abs():max()
   
   mytester:assertlt(err, precision, 'error in torch.add - non contiguous')

   -- [res] torch.add([res,] tensor, value)
   local m1 = torch.randn(10,10)
   local res1 = m1:clone()
   res1[{ 3,{} }]:add(2)
   
   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ 3,i }] = res2[{ 3,i }] + 2
   end
   
   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.add - scalar, contiguous')

   local m1 = torch.randn(10,10)
   local res1 = m1:clone()
   res1[{ {},3 }]:add(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] + 2
   end
   
   local err = (res1-res2):abs():max()

   mytester:assertlt(err, precision, 'error in torch.add - scalar, non contiguous')
   
   -- [res] torch.add([res,] tensor1, value, tensor2)
end

function torchtest.mul()
   local m1 = torch.randn(10,10)
   local res1 = m1:clone()

   res1[{ {},3 }]:mul(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] * 2
   end
   
   local err = (res1-res2):abs():max()
   
   mytester:assertlt(err, precision, 'error in torch.mul - scalar, non contiguous')
end

function torchtest.div()
   local m1 = torch.randn(10,10)
   local res1 = m1:clone()

   res1[{ {},3 }]:div(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] / 2
   end
   
   local err = (res1-res2):abs():max()
   
   mytester:assertlt(err, precision, 'error in torch.div - scalar, non contiguous')
end

function torchtest.pow()  -- [res] torch.pow([res,] x)
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
end

function torchtest.cdiv()  -- [res] torch.cdiv([res,] tensor1, tensor2)
   -- contiguous
   local m1 = torch.randn(10, 10, 10)
   local m2 = torch.randn(10, 10 * 10)
   local sm1 = m1[{4, {}, {}}]
   local sm2 = m2[{4, {}}]
   local res1 = torch.cdiv(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
	 local idx1d = (((i-1)*sm1:size(1)))+j 
	 res2[i][j] = sm1[i][j] / sm2[idx1d]
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
   mytester:assertlt(maxerr, precision, 'error in torch.cdiv - contiguous')

   -- non-contiguous
   local m1 = torch.randn(10, 10, 10)
   local m2 = torch.randn(10 * 10, 10 * 10)
   local sm1 = m1[{{}, 4, {}}]
   local sm2 = m2[{{}, 4}]
   local res1 = torch.cdiv(sm1, sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
	 local idx1d = (((i-1)*sm1:size(1)))+j 
	 res2[i][j] = sm1[i][j] / sm2[idx1d]
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
   mytester:assertlt(maxerr, precision, 'error in torch.cdiv - non-contiguous')  
end

function torchtest.cmul()  -- [res] torch.cmul([res,] tensor1, tensor2)
   -- contiguous
   local m1 = torch.randn(10, 10, 10)
   local m2 = torch.randn(10, 10 * 10)
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
   mytester:assertlt(maxerr, precision, 'error in torch.cmul - contiguous')

   -- non-contiguous
   local m1 = torch.randn(10, 10, 10)
   local m2 = torch.randn(10 * 10, 10 * 10)
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
   mytester:assertlt(maxerr, precision, 'error in torch.cmul - non-contiguous')  
end

function torchtest.sum()
   local x = torch.rand(msize,msize)
   local mx = torch.sum(x,2)
   local mxx = torch.Tensor()
   torch.sum(mxx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sum value')
end
function torchtest.prod()
   local x = torch.rand(msize,msize)
   local mx = torch.prod(x,2)
   local mxx = torch.Tensor()
   torch.prod(mxx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.prod value')
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
   local m2 = renorm(m1,2,2,maxnorm)
   
   m1:renorm(2,2,maxnorm)
   mytester:assertTensorEq(m1, m2, 0.00001)
   mytester:assertTensorEq(m1:norm(2,1), m2:norm(2,1), 0.00001)
   
   m1 = torch.randn(3,4,5)
   m2 = m1:transpose(2,3):contiguous():reshape(15,4)
   
   maxnorm = m2:norm(2,1):mean()
   m2 = renorm(m2,2,2,maxnorm)
   
   m1:renorm(2,2,maxnorm)
   m3 = m1:transpose(2,3):contiguous():reshape(15,4)
   mytester:assertTensorEq(m3, m2, 0.00001)
   mytester:assertTensorEq(m3:norm(2,1), m2:norm(2,1), 0.00001)
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
function torchtest.sortAscending()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.sort(x)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.sort(mxx,ixx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sort (ascending) value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.sort (ascending) index')
   local increasing = true
   for j = 1,msize do
       for k = 2,msize do
           increasing = increasing and (mxx[j][k-1] < mxx[j][k])
       end
   end
   mytester:assert(increasing, 'torch.sort (ascending) increasing')
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
   mytester:assert(indicesCorrect, 'torch.sort (ascending) indices')
   mytester:assertTensorEq(
           torch.sort(torch.Tensor{ 50, 40, 30, 20, 10 }),
           torch.Tensor{ 10, 20, 30, 40, 50 },
           1e-16,
           "torch.sort (ascending) simple sort"
       )
   -- Test that we still have proper sorting with duplicate keys
   local x = torch.floor(torch.rand(msize,msize)*10)
   torch.sort(mxx,ixx,x)
   local increasing = true
   for j = 1,msize do
       for k = 2,msize do
           increasing = increasing and (mxx[j][k-1] <= mxx[j][k])
       end
   end
   mytester:assert(increasing, 'torch.sort (ascending) increasing with equal keys')
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
   mytester:assert(indicesCorrect, 'torch.sort (ascending) indices with equal keys')
end
function torchtest.sortDescending()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.sort(x,true)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.sort(mxx,ixx,x,true)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sort (descending) value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.sort (descending) index')
   local decreasing = true
   for j = 1,msize do
       for k = 2,msize do
           decreasing = decreasing and (mxx[j][k-1] > mxx[j][k])
       end
   end
   mytester:assert(decreasing, 'torch.sort (descending) decreasing')
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
   mytester:assert(indicesCorrect, 'torch.sort (descending) indices')
   mytester:assertTensorEq(
           torch.sort(torch.Tensor{ 10, 20, 30, 40, 50 },true),
           torch.Tensor{ 50, 40, 30, 20, 10 },
           1e-16,
           "torch.sort (descending) simple sort"
       )
   -- Test that we still have proper sorting with duplicate keys
   local x = torch.floor(torch.rand(msize,msize)*10)
   torch.sort(mxx,ixx,x,true)
   local decreasing = true
   for j = 1,msize do
       for k = 2,msize do
           decreasing = decreasing and (mxx[j][k-1] >= mxx[j][k])
       end
   end
   mytester:assert(decreasing, 'torch.sort (descending) decreasing with equal keys')
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
   mytester:assert(indicesCorrect, 'torch.sort (descending) indices with equal keys')
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
   local x = torch.rand(13,msize,msize)
   local y = torch.rand(17,msize,msize)
   local mx = torch.cat(x,y,1)
   local mxx = torch.Tensor()
   torch.cat(mxx,x,y,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cat value')
end
function torchtest.sin()
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
function torchtest.gels()
   if not torch.gels then return end
   local a=torch.Tensor({{ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45},
			 {-7.84, -0.28,  3.24,  8.09,  2.52, -5.70},
			 {-4.39, -3.24,  6.27,  5.28,  0.74, -1.19},
			 {4.53,  3.83, -6.64,  2.06, -2.47,  4.70}}):t()
   local b=torch.Tensor({{8.58,  8.26,  8.48, -5.28,  5.72,  8.93},
			 {9.35, -4.43, -0.70, -0.26, -7.36, -2.52}}):t()
   local mx = torch.gels(b,a)
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gels(tb,ta,b,a)
   local mxxx = torch.gels(b,a,b,a)
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
   uuu,sss,vvv = torch.svd(uu,ss,vv,a)
   mytester:asserteq(maxdiff(u,uu),0,'torch.svd')
   mytester:asserteq(maxdiff(u,uuu),0,'torch.svd')
   mytester:asserteq(maxdiff(s,ss),0,'torch.svd')
   mytester:asserteq(maxdiff(s,sss),0,'torch.svd')
   mytester:asserteq(maxdiff(v,vv),0,'torch.svd')
   mytester:asserteq(maxdiff(v,vvv),0,'torch.svd')
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

function torchtest.TestAsserts()
   mytester:assertError(function() error('hello') end, 'assertError: Error not caught')
   mytester:assertErrorPattern(function() error('hello') end, '.*ll.*', 'assertError: ".*ll.*" Error not caught')

   local x = torch.rand(100,100)*2-1;
   local xx = x:clone();
   mytester:assertTensorEq(x, xx, 1e-16, 'assertTensorEq: not deemed equal')
   mytester:assertTensorNe(x, xx+1, 1e-16, 'assertTensorNe: not deemed different')
   mytester:assertalmosteq(0, 1e-250, 1e-16, 'assertalmosteq: not deemed different')
end

function torchtest.BugInAssertTableEq()
   local t = {1,2,3}
   local tt = {1,2,3}
   mytester:assertTableEq(t, tt, 'assertTableEq: not deemed equal')
   mytester:assertTableNe(t, {3,2,1}, 'assertTableNe: not deemed different')
   mytester:assertTableEq({1,2,{4,5}}, {1,2,{4,5}}, 'assertTableEq: fails on recursive lists')
   mytester:assertTableNe(t, {1,2}, 'assertTableNe: different size not deemed different')
   mytester:assertTableNe(t, {1,2,3,4}, 'assertTableNe: different size not deemed different')
end

function torchtest.RNGState()
   local ignored = torch.rand(1000)
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
    local C = torch.potrf(A)
    local B = torch.mm(C:t(), C)
    mytester:assertTensorEq(A, B, 1e-14, 'potrf did not allow rebuilding the original matrix')
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
   local bignumber = 2^31 + 1
   local input = torch.LongTensor{-bignumber}
   mytester:assertgt(input:abs()[1], 0, 'torch.abs(3)')
end

function torchtest.classInModule()
    -- Need a global for this module
    _mymodule123 = {}
    local x = torch.class('_mymodule123.myclass')
    mytester:assert(x, 'Could not create class in module')
    -- Remove the global
    _G['_mymodule123'] = nil
end

function torchtest.classNoModule()
    local x = torch.class('_myclass123')
    mytester:assert(x, 'Could not create class in module')
end

function torchtest.view()
   local tensor = torch.rand(15)
   local template = torch.rand(3,5)
   local target = template:size():totable()
   mytester:assertTableEq(tensor:viewAs(template):size():totable(), target, 'Error in viewAs')
   mytester:assertTableEq(tensor:view(3,5):size():totable(), target, 'Error in view')
   mytester:assertTableEq(tensor:view(torch.LongStorage{3,5}):size():totable(), target, 'Error in view using LongStorage')
   mytester:assertTableEq(tensor:view(-1,5):size():totable(), target, 'Error in view using dimension -1')
   mytester:assertTableEq(tensor:view(3,-1):size():totable(), target, 'Error in view using dimension -1')
   local tensor_view = tensor:view(5,3)
   tensor_view:fill(torch.rand(1)[1])
   mytester:asserteq((tensor_view-tensor):abs():max(), 0, 'Error in view')

   local target_tensor = torch.Tensor()
   mytester:assertTableEq(target_tensor:viewAs(tensor, template):size():totable(), target, 'Error in viewAs')
   mytester:assertTableEq(target_tensor:view(tensor, 3,5):size():totable(), target, 'Error in view')
   mytester:assertTableEq(target_tensor:view(tensor, torch.LongStorage{3,5}):size():totable(), target, 'Error in view using LongStorage')
   mytester:assertTableEq(target_tensor:view(tensor, -1,5):size():totable(), target, 'Error in view using dimension -1')
   mytester:assertTableEq(target_tensor:view(tensor, 3,-1):size():totable(), target, 'Error in view using dimension -1')
   target_tensor:fill(torch.rand(1)[1])
   mytester:asserteq((target_tensor-tensor):abs():max(), 0, 'Error in viewAs')
end

function torch.test(tests)
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
