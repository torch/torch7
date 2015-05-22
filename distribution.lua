-- Random number generator for a custom discrete distribution.
-- Takes a 1D vector (or array) of probabilites {p_1, p_2, ..., p_n};
-- Returns a generator function gen() which can be used as follows:
--   - gen() returns an integer 1..n; the probability of returning i is
--     proportional to p_i.
--   - gen(...) returns an LongTensor of such random values, e.g.,
--     gen(3,4,5) returns a 3-by-4-by-5 tensor.
--
-- This implements the "alias method", see http://www.keithschwarz.com/darts-dice-coins/
-- Initialization takes O(n), each call to gen() takes O(1).
--
-- Implementation - Mitja Trampus

function torch.distribution(probs)
   -- Cast to tensor, create a copy
   if type(probs) == 'table' then
      probs = torch.FloatTensor(probs)
   else
      probs = probs + 0  -- make a copy
   end
   local n = probs:size(1)
   local alias = torch.LongTensor(n)

   -- Ensure normalization; make the average probability equal to 1
   probs:div(probs:sum()):mul(n)

   -- Build the alias structure
   local small = {}  -- indices of columns that are smaller than average
   local large = {}
   for i = 1,n do
      if probs[i] < 1 then small[#small+1] = i else large[#large+1] = i end
   end
   while #small ~= 0 and #large ~= 0 do
      local l = large[#large]
      local s = small[#small]
      alias[s] = l
      probs[l] = probs[l] - (1 - probs[s])
      small[#small] = nil  -- we're done with this column
      if probs[l] < 1 then
         large[#large] = nil
         small[#small+1] = l
      end
   end
   -- Take care of rounding errors and/or uniform distributions
   for i,s in ipairs(small) do
      alias[s] = s
   end
   for i,l in ipairs(large) do
      alias[l] = l
   end

   -- Return sampling function:
   local cols = torch.LongTensor()
   local seeds = torch.FloatTensor()
   return function(...)
      local size = {...}
      if size[1] then
         local results
         if type(size[1]) == 'number' then
            results = torch.LongTensor(...)
         elseif torch.typename(size[1]):find('torch..*Tensor') then
            results = size[1]
         end
         cols:resize(results:nElement()):random(n)
         seeds:resize(results:nElement()):uniform(0,1)
         local resultsView = results:view(results:nElement())
         for i = 1,resultsView:size(1) do
            local col, seed = cols[i], seeds[i]
            if seed < probs[col] then
               resultsView[i] = col
            else
               resultsView[i] = alias[col]
            end
         end
         return results
      else
         local col = torch.random(n)
         local seed = torch.uniform(0,1)
         if seed < probs[col] then
            return col
         else
            return alias[col]
         end
      end
   end
end
