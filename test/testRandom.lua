local myTester = torch.Tester()

local tests = {}

local function torch_random_pearsons_chi_square(a,b,repeatExperiment, tensorType)
   -- so that the probability of failure of the
   -- Pearson's chi-squared test is < 1e-30
   repeatExperiment = repeatExperiment or 10

   local samples = {}
   local expected = {}
   local tot = b-a+1

   -- Expected range and probabilities
   for i=a,b do
      expected[i] = 1. / tot
   end
   local n = 100000
   if not tensorType then
      for i =1,n do
         local r = torch.random(a,b)
         samples[r] = samples[r] or 0
         samples[r] = samples[r] + 1
      end
   else
      local t = torch[tensorType](n):random(a,b)
      for i =1,n do
         local r = t[i]
         samples[r] = samples[r] or 0
         samples[r] = samples[r] + 1
      end
   end

   -- Verify range equality
   local included = true
   local contains = true

   for k,_ in pairs(samples) do
      if not expected[k] then
         contains = false
      end
   end

   for k,_ in pairs(expected) do
      if not samples[k] then
         included = false
      end
   end
   myTester:assert(contains, 'sampled values must be contained in the expected range ['..a..','..b..']')
   myTester:assert(included, 'sampled values must include the expected range  ['..a..','..b..']')


   -- Compute the statistics
   local cter = 0
   local chiStat = 0
   for k,v in pairs(samples) do
      cter = cter+1
      samples[k] = samples[k] / n
      local val = (samples[k] - expected[k])
      val = val / expected[k]
      val = val*val
      val = expected[k] * val
      chiStat = chiStat + val
   end
   chiStat = chiStat*n

   repeatExperiment = repeatExperiment - 1
   -- 149.99 is the 0.999 p-value of the Pearson's chi-square test
   -- with 100 dimensions
   if chiStat >= 149.449 and repeatExperiment == 0 then
      myTester:assert(contains, "Never satisfied the Pearson's chi-square"..
                                 "test after 10 experiments for a: "..a.." and b: "..b)
   elseif chiStat >= 149.449 then
      torch_random_pearsons_chi_square(a,b,repeatExperiment)
   end
end

function tests.randomDistribution()
   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10)

   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10,'IntTensor')

   local a=2^53-100
   local b=2^53-1
   torch_random_pearsons_chi_square(a,b,10)

   local a=2^32-50
   local b=2^32+49
   torch_random_pearsons_chi_square(a,b,10)
end


function tests.randomDistributionCharTensor()
   -- Test for multiple values of a and b
   local a=-128
   local b=-27
   torch_random_pearsons_chi_square(a,b,10,'CharTensor')

   local a=28
   local b=127
   torch_random_pearsons_chi_square(a,b,10, 'CharTensor')
end

function tests.randomDistributionByteTensor()
   -- Test for multiple values of a and b
   local a=0
   local b=99
   torch_random_pearsons_chi_square(a,b,10,'ByteTensor')

   local a=156
   local b=255
   torch_random_pearsons_chi_square(a,b,10, 'ByteTensor')
end

function tests.randomDistributionShortTensor()
   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10,'ShortTensor')

   local a=2^15-100
   local b=2^15-1
   torch_random_pearsons_chi_square(a,b,10, 'ShortTensor')
end

function tests.randomDistributionIntTensor()
   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10,'IntTensor')

   local a=2^31-100
   local b=2^31-1
   torch_random_pearsons_chi_square(a,b,10, 'IntTensor')
end

function tests.randomDistributionFloatTensor()
   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10,'FloatTensor')

   local a=2^24-100
   local b=2^24-1
   torch_random_pearsons_chi_square(a,b,10, 'FloatTensor')
end

function tests.randomDistributionDoubleTensor()
   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10,'DoubleTensor')

   local a=2^53-100
   local b=2^53-1
   torch_random_pearsons_chi_square(a,b,10, 'DoubleTensor')
end


function tests.randomDistributionLongTensor()
   -- Test for multiple values of a and b
   local a=-50
   local b=49
   torch_random_pearsons_chi_square(a,b,10,'LongTensor')

   -- Can't go over 2^53 because of the multiplication in random...
   local a=2^53-100
   local b=2^53-1
   torch_random_pearsons_chi_square(a,b,10, 'LongTensor')
end

myTester:add(tests)
myTester:run()
