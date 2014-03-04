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

function torchtest.max()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.max(x,1)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.max(mxx,ixx,x,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.max value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.max index')
end
function torchtest.min()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.min(x,2)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.min(mxx,ixx,x,2)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.min value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.min index')
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
   mytester:assertlt(math.abs(x:dot(x)-torch.xcorr3(x,x)[1][1][1]),1e-10,'torch.conv3')

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
   mytester:assertErrorMsg(function() error('hello') end, 'test.lua:482: hello', 'assertError: "hello" Error not caught')
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

   mytester:assert(state:ne(stateCloned):long():sum() == 0, 'RNG (supposedly cloned) state has changed after random number generation')

   torch.setRNGState(state)
   local after = torch.rand(1000)
   mytester:assertTensorEq(before, after, 1e-16, 'getRNGState/setRNGState not generating same sequence')
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

function torchtest.testCholesky()
    local x = torch.rand(10,10)
    local A = torch.mm(x, x:t())
    local C = torch.potrf(A)
    local B = torch.mm(C:t(), C)
    mytester:assertTensorEq(A, B, 1e-14, 'potrf did not allow rebuilding the original matrix')
end
function torchtest.testCholeskyErrsOnRankDeficient()
    local x = torch.rand(5,5)
    local A = torch.mm(x, x:t())
    -- Make the matrix rank-defficient
    A[{{},5}]:copy(A[{{},{1}}])
    mytester:assertError(function() torch.potrf(A) end)
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

function torch.test()
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(torchtest)
   mytester:run()
end
