local Tester = torch.class('torch.Tester')

function Tester:__init()
   self.errors = {}
   self.tests = {}
   self.testnames = {}
   self.curtestname = ''
end


function Tester:assert_sub (condition, message)
   self.countasserts = self.countasserts + 1
   if not condition then
      local ss = debug.traceback('tester',2)
      --print(ss)
      ss = ss:match('[^\n]+\n[^\n]+\n([^\n]+\n[^\n]+)\n')
      self.errors[#self.errors+1] = self.curtestname .. '\n' .. (message or '') .. '\n' .. ss .. '\n'
   end
end

function Tester:assert (condition, message)
   self:assert_sub(condition,string.format('%s\n%s  condition=%s',(message or ''),' BOOL violation ', tostring(condition)))
end

function Tester:assertlt (val, condition, message)
   self:assert_sub(val<condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' LT(<) violation ', tostring(val), tostring(condition)))
end

function Tester:assertgt (val, condition, message)
   self:assert_sub(val>condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' GT(>) violation ', tostring(val), tostring(condition)))
end

function Tester:assertle (val, condition, message)
   self:assert_sub(val<=condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' LE(<=) violation ', tostring(val), tostring(condition)))
end

function Tester:assertge (val, condition, message)
   self:assert_sub(val>=condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' GE(>=) violation ', tostring(val), tostring(condition)))
end

function Tester:asserteq (val, condition, message)
   self:assert_sub(val==condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' EQ(==) violation ', tostring(val), tostring(condition)))
end

function Tester:assertalmosteq (a, b, condition, message)
   condition = condition or 1e-16
   local err = math.abs(a-b)
   self:assert_sub(err < condition, string.format('%s\n%s  val=%s, condition=%s',(message or ''),' ALMOST_EQ(==) violation ', tostring(err), tostring(condition)))
end

function Tester:assertne (val, condition, message)
   self:assert_sub(val~=condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' NE(~=) violation ', tostring(val), tostring(condition)))
end

function Tester:assertTensorEq(ta, tb, condition, message)
   local diff = ta-tb
   local err = diff:abs():max()
   self:assert_sub(err<condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' TensorEQ(==) violation ', tostring(err), tostring(condition)))
end

function Tester:assertTensorNe(ta, tb, condition, message)
   local diff = ta-tb
   local err = diff:abs():max()
   self:assert_sub(err>=condition,string.format('%s\n%s  val=%s, condition=%s',(message or ''),' TensorNE(~=) violation ', tostring(err), tostring(condition)))
end

local function areTablesEqual(ta, tb)
   local function isIncludedIn(ta, tb)
      if type(ta) ~= 'table' or type(tb) ~= 'table' then
         return ta == tb
      end
      for k, v in pairs(tb) do
         if not areTablesEqual(ta[k], v) then return false end
      end
      return true
   end

   return isIncludedIn(ta, tb) and isIncludedIn(tb, ta)
end

function Tester:assertTableEq(ta, tb, message)
   self:assert_sub(areTablesEqual(ta, tb), string.format('%s\n%s',(message or ''),' TableEQ(==) violation '))
end

function Tester:assertTableNe(ta, tb, message)
   self:assert_sub(not areTablesEqual(ta, tb), string.format('%s\n%s',(message or ''),' TableEQ(==) violation '))
end

function Tester:assertError(f, message)
    return self:assertErrorObj(f, function(err) return true end, message)
end

function Tester:assertErrorMsg(f, errmsg, message)
    return self:assertErrorObj(f, function(err) return err == errmsg end, message)
end

function Tester:assertErrorPattern(f, errPattern, message)
    return self:assertErrorObj(f, function(err) return string.find(err, errPattern) ~= nil end, message)
end

function Tester:assertErrorObj(f, errcomp, message)
    -- errcomp must be  a function  that compares the error object to its expected value
   local status, err = pcall(f)
   self:assert_sub(status == false and errcomp(err), string.format('%s\n%s  err=%s', (message or ''),' ERROR violation ', tostring(err)))
end



function Tester:pcall(f)
   local nerr = #self.errors
   -- local res = f()
   local stat, result = xpcall(f, debug.traceback)
   if not stat then
      self.errors[#self.errors+1] = self.curtestname .. '\n Function call failed \n' .. result .. '\n'
   end
   return stat, result, stat and (nerr == #self.errors)
   -- return true, res, nerr == #self.errors
end

function Tester:report(tests)
   if not tests then
      tests = self.tests
   end
   print('Completed ' .. self.countasserts .. ' asserts in ' .. #tests .. ' tests with ' .. #self.errors .. ' errors')
   print()
   print(string.rep('-',80))
   for i,v in ipairs(self.errors) do
      print(v)
      print(string.rep('-',80))
   end
end

function Tester:run(run_tests)
   local tests, testnames
   self.countasserts = 0
   tests = self.tests
   testnames = self.testnames
   if type(run_tests) == 'string' then
      run_tests = {run_tests}
   end
   if type(run_tests) == 'table' then
      tests = {}
      testnames = {}
      for i,fun in ipairs(self.tests) do
         for j,name in ipairs(run_tests) do
            if self.testnames[i] == name or i == name then
               tests[#tests+1] = self.tests[i]
               testnames[#testnames+1] = self.testnames[i]
            end
         end
      end
   end

   print('Running ' .. #tests .. ' tests')
   local statstr = string.rep('_',#tests)
   local pstr = ''
   io.write(statstr .. '\r')
   for i,v in ipairs(tests) do
      self.curtestname = testnames[i]

      --clear
      io.write('\r' .. string.rep(' ', pstr:len()))
      io.flush()
      --write
      pstr = statstr:sub(1,i-1) .. '|' .. statstr:sub(i+1) .. '  ==> ' .. self.curtestname
      io.write('\r' .. pstr)
      io.flush()

      local stat, message, pass = self:pcall(v)

      if pass then
         --io.write(string.format('\b_'))
         statstr = statstr:sub(1,i-1) .. '_' .. statstr:sub(i+1)
      else
         statstr = statstr:sub(1,i-1) .. '*' .. statstr:sub(i+1)
         --io.write(string.format('\b*'))
      end

      if not stat then
         -- print()
         -- print('Function call failed: Test No ' .. i .. ' ' .. testnames[i])
         -- print(message)
      end
      collectgarbage()
   end
   --clear
   io.write('\r' .. string.rep(' ', pstr:len()))
   io.flush()
   -- write finish
   pstr = statstr .. '  ==> Done '
   io.write('\r' .. pstr)
   io.flush()
   print()
   print()
   self:report(tests)
end

function Tester:add(f,name)
   name = name or 'unknown'
   if type(f) == "table" then
      for i,v in pairs(f) do
         self:add(v,i)
      end
   elseif type(f) == "function" then
      self.tests[#self.tests+1] = f
      self.testnames[#self.tests] = name
   else
      error('Tester:add(f) expects a function or a table of functions')
   end
end
