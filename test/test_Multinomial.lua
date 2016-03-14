-- Test multinomial for rare events (based on https://github.com/torch/torch7/issues/418)
-- and for performance (cf. https://github.com/torch/torch7/issues/453)

sys.tic()
do
   local p = torch.FloatTensor(1001000):fill(1)
   p:narrow(1, 50001, 50000):fill(1e-3)
   p:div(p:sum())
   local N = 1001000

   local n = 0
   local c = torch.LongTensor(p:nElement()):zero()
   local c_ptr = c:data() - 1
   local tmp = torch.LongTensor()
   for i = 1, 100 do
      p.multinomial(tmp, p, N, true);
      n = n + N
      tmp:apply(function(i) c_ptr[i] = c_ptr[i] + 1 end)
   end

   local actual = c:narrow(1, 50001, 50000):sum()
   local expected = n*p:narrow(1, 50001, 50000):sum()
   print('Actual, Expected: ', actual, expected)
end
print('Time spent: ', sys.toc())
