local tester = torch.Tester()


local function aliasMultinomial()
   local n_class = 10000
   local probs = torch.Tensor(n_class):uniform(0,1)
   probs:div(probs:sum())
   local a = torch.Timer()
   local state = torch.multinomialAliasSetup(probs)
   print("AliasMultinomial setup in "..a:time().real.." seconds(hot)")
   a:reset()
   state = torch.multinomialAliasSetup(probs, state)
   print("AliasMultinomial setup in "..a:time().real.." seconds(cold)")
   a:reset()
   
   tester:assert(state[1]:min() >= 0, "Index ="..state[1]:min().."alias indices has an index below or equal to 0")
   tester:assert(state[1]:max() <= n_class, state[1]:max().." alias indices has an index exceeding num_class")
   local output = torch.LongTensor(1000000)
   torch.multinomialAlias(output, state)
   local n_samples = output:nElement()
   print("AliasMultinomial draw "..n_samples.." elements from "..n_class.." classes ".."in "..a:time().real.." seconds")
   local counts = torch.Tensor(n_class):zero()
   mult_output = torch.multinomial(probs, n_samples, true)
   print("Multinomial draw "..n_samples.." elements from "..n_class.." classes ".." in "..a:time().real.." seconds")
   tester:assert(output:min() > 0, "sampled indices has an index below or equal to 0")
   tester:assert(output:max() <= n_class, "indices has an index exceeding num_class")
   output:apply(function(x)
         counts[x] = counts[x] + 1
   end)
   a:reset()
   
   counts:div(counts:sum())
   
   tester:assert(state[1]:min() >= 0, "Index ="..state[1]:min().."alias indices has an index below or equal to 0")
   tester:assert(state[1]:max() <= n_class, state[1]:max().." alias indices has an index exceeding num_class")
   tester:eq(probs, counts, 0.001, "probs and counts should be approximately equal")
end

tester:add(aliasMultinomial)
tester:run()
