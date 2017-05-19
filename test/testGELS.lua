-- Test linear regression with m observations, and dimension n
-- Note: in statistics and linear regression, the number of obs would be 'n'
-- and the dimension 'p', but I stick to torch and lapack's conventions

require 'torch'
require 'totem'

local tester = totem.Tester()
local myTests = {}


local verbose = false
local verbose, verbose
if verbose then
  verbose = print
else
  verbose = function() end
end


-- Regression using torch.gels
local function regGels(X, Y)
  local theta = torch.gels(Y, X)
  return theta
end


-- Regression reimplemented with pseudo inverse and regularization
local function regPseudoInv(X, Y)
  local nObs = X:size(1)
  local inputSize = X:size(2)

  -- Actually useful
  local theta = torch.Tensor(inputSize)

  -- Temporaries
  local xtx = torch.Tensor(inputSize, inputSize)
  local pseudoInv = torch.Tensor(inputSize, inputSize)
  local pseudoInvLU = torch.Tensor(inputSize, inputSize)
  local predBatch = torch.Tensor()


  xtx:mm(X:t(), X)
  -- Deal with undetermined case using regularization
  if nObs < inputSize then
    local epsilon = 1e-9
    xtx:add(torch.eye(inputSize):mul(epsilon))
  end

  torch.gesv(pseudoInv, pseudoInvLU, X:t(), xtx)
  if Y:dim() > 1 then
    assert(Y:size(2) == 1, 'Sorry, only dealing with single prediction for now')
    Y = Y:view(Y:size(1))
  end
  theta:mv(pseudoInv, Y)

  return theta:view(theta:size(1), 1)
end


--[[ opts contains
m
n
reg: function actually doing the regression
]]
local function testDoesNotCrash(opts)
  -- We have n observations, each of dimension p
  local m = opts.m
  local n = opts.n
  verbose('----------------')
  verbose('m:', m, 'n:', n)

  -- Choose a regression vector
  local theta = torch.rand(n, 1)
  verbose('theta')
  verbose(theta)

  -- Get the observations, making sure that a is full rank
  local a = torch.eye(m, n)
  verbose('a is m-by-n')
  verbose(a)

  -- And compute the true result, without noise to allow perfect match
  local b = a * theta

  verbose('b')
  verbose(b)

  -- Actual regression
  local x = opts.reg(a, b)

  verbose('x:')
  verbose(x)
  verbose('theta')
  verbose(theta)
  verbose('a * x:narrow')
  verbose(a * x:narrow(1, 1, n))
  verbose('b')
  verbose(b)

  verbose(b:dist(a*x:narrow(1, 1, n)))
  tester:assertTensorEq(b,
                        a*x:narrow(1, 1, n),
                        1e-9, 'Did not recover exact')

  if x:size(1) > n then
    verbose(math.sqrt(x:narrow(1, n + 1, x:size(1) - n):pow(2):sum()))
  else
    verbose('no extra components for residuals')
  end
end


local function testResultsMatch(opts)
  -- We have n observations, each of dimension p
  local m = opts.m
  local n = opts.n

  -- Choose a regression vector
  local theta = torch.rand(n, 1)

  -- Get the observations, making sure that a is full rank
  local a = torch.eye(m, n)

  -- And compute the true result
  local b = a * theta + torch.randn(m, 1)

  -- Actual regression
  local xgels = regGels(a, b)
  local xpseudoinv = regPseudoInv(a, b)

  if xgels:size(1) > n then
    xgels = xgels:narrow(1, 1, n)
  end

  verbose(xgels)
  verbose(xpseudoinv)
  tester:assertTensorEq(xgels, xpseudoinv, 1e-9,
                        'Regression results do not match')
end



for k, v in pairs({GELS = regGels, Home = regPseudoInv}) do
  myTests['test' .. k .. 'OverDetermined']  = function()
    -- PASS: more observations than dimensions
    testDoesNotCrash{m = 5, n = 4, reg = v}
  end

  -- PASS: same number of observations as dimensions
  myTests['test' .. k .. 'ExactDetermined']  = function()
    testDoesNotCrash{m = 4, n = 4, reg = v}
  end

  -- FAIL: less observations than dimension, i.e. m < n
  myTests['test' .. k .. 'UnderDetermined']  = function()
    local success, err = pcall(testDoesNotCrash, {m = 3, n = 4, reg = v})
    tester:assert(success, 'Crashed with error ' .. tostring(err))
  end
end

myTests.testMatchOverDetermined = function() testResultsMatch{m = 5, n = 4} end
myTests.testMatchExactDetermined = function() testResultsMatch{m = 4, n = 4} end
myTests.testMatchUnderDetermined = function()
  local success, err = pcall(testResultsMatch, {m = 3, n = 4})
  tester:assert(success, 'Crashed with error ' .. tostring(err))
end

return tester:add(myTests):run()
