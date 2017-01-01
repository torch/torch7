
-- Lua 5.2 compatibility
local unpack = unpack or table.unpack

local check = {} -- helper functions, defined at the bottom of the file

local Tester = torch.class('torch.Tester')

function Tester:__init()
   self.errors = {}
   self.tests = {}
   self.warnings = {}
   self._warningCount = {}
   self.disabledTests = {}
   self._currentTestName = ''

   -- To maintain backwards compatibility (at least for a short while),
   -- disable exact dimension checking of tensors when :assertTensorEq is
   -- called. Thus {{1}} == {1} when this flag is true.
   --
   -- Note that other methods that suppose tensor checking (such as
   -- :assertGeneralEq) ignore this flag, since previously they didn't
   -- exist or support tensor equality checks at all, so there is no
   -- old code that uses these functions and relies on the behaviour.
   --
   -- Note also that if the dimension check fails with this flag is true, then
   -- will show a warning.
   self._assertTensorEqIgnoresDims = true
end

function Tester:setEarlyAbort(earlyAbort)
   self.earlyAbort = earlyAbort
end

function Tester:setRethrowErrors(rethrow)
   self.rethrow = rethrow
end

function Tester:setSummaryOnly(summaryOnly)
   self.summaryOnly = summaryOnly
end

-- Add a success to the test.
function Tester:_success()
   local name = self._currentTestName
   self.assertionPass[name] = self.assertionPass[name] + 1
   return true
end

function Tester:_addDebugInfo(message)
   local ss = debug.traceback('tester', 3) or ''
   ss = ss:match('.-\n([^\n]+\n[^\n]+)\n[^\n]+xpcall') or ''
   local name = self._currentTestName
   return (name ~= '' and name .. '\n' or '') .. message .. '\n' .. ss
end

-- Add a failure to the test.
function Tester:_failure(message)
   if self.rethrow then error(message, 2) end
   local name = self._currentTestName
   self.assertionFail[name] = self.assertionFail[name] + 1
   self.errors[#self.errors + 1] = self:_addDebugInfo(message)
   return false
end

-- Add a warning to the test
function Tester:_warning(message)
   local name = self._currentTestName
   self._warningCount[name] = (self._warningCount[name] or 0) + 1
   self.warnings[#self.warnings + 1] = self:_addDebugInfo(message)
end

-- Call this during a test run with `condition = true` to log a success, or with
-- `condition = false` to log a failure (using `message`).
function Tester:_assert_sub(condition, message)
   if condition then
      return self:_success()
   else
      return self:_failure(message)
   end
end

local function getMessage(message, ...)
   assert(next{...} == nil, "Unexpected arguments passed to test function")
   if message then
      assert(type(message) == 'string', 'message parameter must be a string')
      if message ~= '' then
         return message .. '\n'
      end
   end
   return ''
end

--[[ Historically, some test functions have accepted both a message and a
tolerance, and some just a message (e.g., assertTableEq). Now assertTableEq
accepts both a tolerance and a message, so allow the two arguments to be passed
in either order to maintain backwards compatibility (and more generally,
for convenience). (We still document the ordering as "tolerance, message" for
clarity.) This function also sanitizes them (ensures they are non-nil, etc).
]]
local function getToleranceAndMessage(defaultTolerance, ...)
   local args = {...}
   local message = nil
   local tolerance = nil
   for _, a in ipairs(args) do
      if type(a) == 'string' then
         if message then
            error("Unexpected string argument; already have message", a)
         end
         message = a .. '\n'
      elseif type(a) == 'number' then
         if tolerance then
            error("Unexpected number argument; already have tolerance", a)
         end
         tolerance = a
         assert(tolerance >= 0, "tolerance cannot be negative")
      else
         error("Unrecognized argument; should be a tolerance or message", a)
      end
   end
   message = message or ''
   tolerance = tolerance or defaultTolerance
   return tolerance, message
end

function Tester:assert(condition, ...)
   local message = getMessage(...)
   if type(condition) ~= 'boolean' then
      self:_warning(" :assert should only be used for boolean conditions. "
                    .. "To check for non-nil variables, do this explicitly: "
                    .. "Tester:assert(var ~= nil).")
   end
   return self:_assert_sub(condition,
                           string.format('%sBOOL violation condition=%s',
                                         message, tostring(condition)))
end

function Tester:assertGeneralEq(got, expected, ...)
   return self:_eqOrNeq(got, expected, false, ...)
end

function Tester:eq(got, expected, ...)
   return self:assertGeneralEq(got, expected, ...)
end

function Tester:assertGeneralNe(got, unexpected, ...)
   return self:_eqOrNeq(got, unexpected, true, ...)
end

function Tester:ne(got, unexpected, ...)
   return self:assertGeneralNe(got, unexpected, ...)
end

function Tester:_eqOrNeq(got, expected, negate, ...)
   local tolerance, message = getToleranceAndMessage(0, ...)
   local success, subMessage = check.areEq(got, expected, tolerance, negate)
   subMessage = subMessage or ''
   return self:_assert_sub(success, message .. subMessage)
end

function Tester:assertlt(a, b, ...)
   local message = getMessage(...)
   return self:_assert_sub(a < b,
                           string.format('%sLT failed: %s >= %s',
                                         message, tostring(a), tostring(b)))
end

function Tester:assertgt(a, b, ...)
   local message = getMessage(...)
   return self:_assert_sub(a > b,
                           string.format('%sGT failed: %s <= %s',
                                         message, tostring(a), tostring(b)))
end

function Tester:assertle(a, b, ...)
   local message = getMessage(...)
   return self:_assert_sub(a <= b,
                           string.format('%sLE failed: %s > %s',
                                         message, tostring(a), tostring(b)))
end

function Tester:assertge(a, b, ...)
   local message = getMessage(...)
   return self:_assert_sub(a >= b,
                           string.format('%sGE failed: %s < %s',
                                         message, tostring(a), tostring(b)))
end

function Tester:assertalmosteq(a, b, ...)
   local tolerance, message = getToleranceAndMessage(1e-16, ...)
   local diff = math.abs(a - b)
   return self:_assert_sub(
         diff <= tolerance,
         string.format(
               '%sALMOST_EQ failed: %s ~= %s with tolerance=%s',
               message, tostring(a), tostring(b), tostring(tolerance)))
end

function Tester:asserteq(a, b, ...)
   local message = getMessage(...)
   return self:_assert_sub(a == b,
                           string.format('%sEQ failed: %s ~= %s',
                                         message, tostring(a), tostring(b)))
end

function Tester:assertne(a, b, ...)
   local message = getMessage(...)
   if type(a) == type(b) and type(a) == 'table' or type(a) == 'userdata' then
      self:_warning(" :assertne should only be used to compare basic lua "
                    .. "objects (numbers, booleans, etc). Consider using "
                    .. "either :assertGeneralNe or :assert(a ~= b).")
   end
   return self:_assert_sub(a ~= b,
                           string.format('%sNE failed: %s == %s',
                                         message, tostring(a), tostring(b)))
end

function Tester:assertTensorEq(ta, tb, ...)
  return self:_assertTensorEqOrNeq(ta, tb, false, ...)
end

function Tester:assertTensorNe(ta, tb, ...)
  return self:_assertTensorEqOrNeq(ta, tb, true, ...)
end

function Tester:_assertTensorEqOrNeq(ta, tb, negate, ...)
   assert(torch.isTensor(ta), "First argument should be a Tensor")
   assert(torch.isTensor(tb), "Second argument should be a Tensor")

   local tolerance, message = getToleranceAndMessage(0, ...)
   local success, subMessage =
         check.areTensorsEq(ta, tb, tolerance, negate,
                            self._assertTensorEqIgnoresDims)
   subMessage = subMessage or ''

   if self._assertTensorEqIgnoresDims and (not negate) and success
         and not ta:isSameSizeAs(tb) then
     self:_warning("Tensors have the same content but different dimensions. "
                   .. "For backwards compatibility, they are considered equal, "
                   .. "but this may change in the future. Consider using :eq "
                   .. "to check for equality instead.")
   end

   return self:_assert_sub(success, message .. subMessage)
end

function Tester:assertTableEq(ta, tb, ...)
   return self:_assertTableEqOrNeq(ta, tb, false, ...)
end

function Tester:assertTableNe(ta, tb, ...)
   return self:_assertTableEqOrNeq(ta, tb, true, ...)
end

function Tester:_assertTableEqOrNeq(ta, tb, negate, ...)
   assert(type(ta) == 'table', "First argument should be a Table")
   assert(type(tb) == 'table', "Second argument should be a Table")
   return self:_eqOrNeq(ta, tb, negate, ...)
end

function Tester:assertError(f, ...)
   return self:assertErrorObj(f, function() return true end, ...)
end

function Tester:assertNoError(f, ...)
   local message = getMessage(...)
   local status, err = pcall(f)
   return self:_assert_sub(status,
                           string.format('%sERROR violation: err=%s', message,
                                         tostring(err)))
end

function Tester:assertErrorMsg(f, errmsg, ...)
   return self:assertErrorObj(f, function(err) return err == errmsg end, ...)
end

function Tester:assertErrorPattern(f, errPattern, ...)
   local function errcomp(err)
      return string.find(err, errPattern) ~= nil
   end
   return self:assertErrorObj(f, errcomp, ...)
end

function Tester:assertErrorObj(f, errcomp, ...)
   local message = getMessage(...)
   local status, err = pcall(f)
   return self:_assert_sub((not status) and errcomp(err),
                           string.format('%sERROR violation: err=%s', message,
                                         tostring(err)))
end

function Tester:add(f, name)
   if type(f) == "table" then
      assert(name == nil, "Name parameter is forbidden for a table of tests, "
                          .. "since its use is ambiguous")
      if f.__isTestSuite then
         f = f.__tests
      else
         self:_warning("Should use TestSuite rather than plain lua table")
      end
      for i, v in pairs(f) do
         -- We forbid nested tests because the "expected" behaviour when a named
         -- test is run in the case that the named test is in fact a table of
         -- tests is not supported. Similar issue with _setUp and _tearDown
         -- functions inside nested tests.
         assert(type(v) ~= 'table', "Nested sets of tests are not supported")
         self:add(v, i)
      end
      return self
   end

   assert(type(f) == 'function',
          "Only tables of functions and functions supported")

   if name == '_setUp' then
      assert(not self._setUp, "Only one set-up function allowed")
      self._setUp = f
   elseif name == '_tearDown' then
      assert(not self._tearDown, "Only one tear-down function allowed")
      self._tearDown = f
   else
      name = name or 'unknown'
      if self.tests[name] ~= nil then
         error('Test with name ' .. name .. ' already exists!')
      end
      self.tests[name] = f
   end
   return self
end

function Tester:disable(testNames)
   if type(testNames) == 'string' then
      testNames = {testNames}
   end
   assert(type(testNames) == 'table', "Expecting name or list for disable")
   for _, name in ipairs(testNames) do
      assert(self.tests[name], "Unrecognized test '" .. name .. "'")
      self.disabledTests[name] = true
   end
   return self
end

function Tester:run(testNames)
   local tests = self:_getTests(testNames)
   self.assertionPass = {}
   self.assertionFail = {}
   self.haveWarning = {}
   self.testError = {}
   for name in pairs(tests) do
      self.assertionPass[name] = 0
      self.assertionFail[name] = 0
      self.testError[name] = 0
      self._warningCount[name] = 0
   end
   self:_run(tests)
   self:_report(tests)

   -- Throws an error on test failure/error, so that test script returns
   -- with nonzero return value.
   for name in pairs(tests) do
      assert(self.assertionFail[name] == 0,
             'An error was found while running tests!')
      assert(self.testError[name] == 0,
             'An error was found while running tests!')
   end

   return 0
end

local function pluralize(num, str)
   local stem = num .. ' ' .. str
   if num == 1 then
      return stem
   else
      return stem .. 's'
   end
end

local NCOLS = 80
local coloured
local enable_colors, c = pcall(require, 'sys.colors')
if arg and enable_colors then  -- have we been invoked from the commandline?
   coloured = function(str, colour)
      return colour .. str .. c.none
   end
else
   c = {}
   coloured = function(str)
      return str
   end
end

function Tester:_run(tests)
   local ntests = 0
   for _ in pairs(tests) do
      ntests = ntests + 1
   end

   local ntestsAsString = string.format('%u', ntests)
   local cfmt = string.format('%%%uu/%u ', ntestsAsString:len(), ntestsAsString)
   local cfmtlen = ntestsAsString:len() * 2 + 2

   local function bracket(str)
      return '[' .. str .. ']'
   end

   io.write('Running ' .. pluralize(ntests, 'test') .. '\n')
   local i = 1
   for name, fn in pairs(tests) do
      self._currentTestName = name

      -- TODO: compute max length of name and cut it down to size if needed
      local strinit = coloured(string.format(cfmt, i), c.cyan)
                      .. self._currentTestName .. ' '
                      .. string.rep('.',
                                    NCOLS - 6 - 2 -
                                    cfmtlen - self._currentTestName:len())
                      .. ' '
      io.write(strinit .. bracket(coloured('WAIT', c.cyan)))
      io.flush()

      local status, message, pass, skip
      if self.disabledTests[name] then
         skip = true
      else
         skip = false
         if self._setUp then
            self._setUp(name)
         end
         if self.rethrow then
            status = true
            local nerr = #self.errors
            message = fn()
            pass = nerr == #self.errors
         else
            status, message, pass = self:_pcall(fn)
         end
         if self._tearDown then
            self._tearDown(name)
         end
      end

      io.write('\r')
      io.write(strinit)

      if skip then
         io.write(bracket(coloured('SKIP', c.yellow)))
      elseif not status then
         self.testError[name] = 1
         io.write(bracket(coloured('ERROR', c.magenta)))
      elseif not pass then
         io.write(bracket(coloured('FAIL', c.red)))
      else
         io.write(bracket(coloured('PASS', c.green)))
         if self._warningCount[name] > 0 then
            io.write('\n' .. string.rep(' ', NCOLS - 10))
            io.write(bracket(coloured('+warning', c.yellow)))
         end
      end
      io.write('\n')
      io.flush()

      if self.earlyAbort and (i < ntests) and (not status or not pass)
            and (not skip) then
         io.write('Aborting on first error, not all tests have been executed\n')
         break
      end

      i = i + 1

      collectgarbage()
   end
end

function Tester:_pcall(f)
   local nerr = #self.errors
   local stat, result = xpcall(f, debug.traceback)
   if not stat then
      self.errors[#self.errors + 1] =
         self._currentTestName .. '\n Function call failed\n' .. result .. '\n'
   end
   return stat, result, stat and (nerr == #self.errors)
end

function Tester:_getTests(testNames)
   if testNames == nil then
      return self.tests
   end
   if type(testNames) == 'string' then
      testNames = {testNames}
   end
   assert(type(testNames) == 'table',
          "Only accept a name or table of test names (or nil for all tests)")

   local function getMatchingNames(pattern)
      local matchingNames = {}
      for name in pairs(self.tests) do
         if string.match(name, pattern) then
            table.insert(matchingNames, name)
         end
      end
      return matchingNames
   end

   local tests = {}
   for _, pattern in ipairs(testNames) do
      local matchingNames = getMatchingNames(pattern)
      assert(#matchingNames > 0, "Couldn't find test '" .. pattern .. "'")
      for _, name in ipairs(matchingNames) do
         tests[name] = self.tests[name]
      end
   end
   return tests
end

function Tester:_report(tests)
   local ntests = 0
   local nfailures = 0
   local nerrors = 0
   local nskipped = 0
   local nwarnings = 0
   self.countasserts = 0
   for name in pairs(tests) do
      ntests = ntests + 1
      self.countasserts = self.countasserts + self.assertionFail[name]
                          + self.assertionPass[name]
      if self.assertionFail[name] > 0 then
         nfailures = nfailures + 1
      end
      if self.testError[name] > 0 then
         nerrors = nerrors + 1
      end
      if self._warningCount[name] > 0 then
         nwarnings = nwarnings + 1
      end
      if self.disabledTests[name] then
         nskipped = nskipped + 1
      end
   end
   if self._warningCount[''] then
      nwarnings = nwarnings + self._warningCount['']
   end

   io.write('Completed ' .. pluralize(self.countasserts, 'assert'))
   io.write(' in ' .. pluralize(ntests, 'test') .. ' with ')
   io.write(coloured(pluralize(nfailures, 'failure'),
                     nfailures == 0 and c.green or c.red))
   io.write(' and ')
   io.write(coloured(pluralize(nerrors, 'error'),
                     nerrors == 0 and c.green or c.magenta))
   if nwarnings > 0 then
      io.write(' and ')
      io.write(coloured(pluralize(nwarnings, 'warning'), c.yellow))
   end
   if nskipped > 0 then
      io.write(' and ')
      io.write(coloured(nskipped .. ' disabled', c.yellow))
   end
   io.write('\n')

   -- Prints off a message separated by -----
   local haveSection = false
   local function addSection(text)
      local function printDashes()
         io.write(string.rep('-', NCOLS) .. '\n')
      end
      if not haveSection then
         printDashes()
         haveSection = true
      end
      io.write(text .. '\n')
      printDashes()
   end

   if not self.summaryOnly then
      for _, v in ipairs(self.errors) do
         addSection(v)
      end
      for _, v in ipairs(self.warnings) do
         addSection(v)
      end
   end
end


--[[ Tests for tensor equality between two tensors of matching sizes and types.

Tests whether the maximum element-wise difference between `ta` and `tb` is less
than or equal to `tolerance`.

Arguments:
* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (number) maximum elementwise difference between `ta` and `tb`.
* `negate` (boolean) if true, we invert success and failure.
* `storage` (boolean) if true, we print an error message referring to Storages
    rather than Tensors.

Returns:
1. success, boolean that indicates success
2. failure_message, string or nil
]]
function check.areSameFormatTensorsEq(ta, tb, tolerance, negate, storage)
   local function ensureHasAbs(t)
      -- Byte, Char and Short Tensors don't have abs
      return t.abs and t or t:double()
   end

   ta = ensureHasAbs(ta)
   tb = ensureHasAbs(tb)

   local diff = ta:clone():add(-1, tb):abs()
   local err = diff:max()
   local success = err <= tolerance
   if negate then
      success = not success
   end

   local errMessage
   if not success then
      local prefix = storage and 'Storage' or 'Tensor'
      local violation = negate and 'NE(==)' or 'EQ(==)'
      errMessage = string.format('%s%s violation: max diff=%s, tolerance=%s',
                                 prefix,
                                 violation,
                                 tostring(err),
                                 tostring(tolerance))
   end

   return success, errMessage
end

--[[ Tests for tensor equality.

Tests whether the maximum element-wise difference between `ta` and `tb` is less
than or equal to `tolerance`.

Arguments:
* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (number) maximum elementwise difference between `ta` and `tb`.
* `negate` (boolean) if negate is true, we invert success and failure.
* `ignoreTensorDims` (boolean, default false) if true, then tensors of the same
    size but different dimensions can still be considered equal, e.g.,
    {{1}} == {1}. For backwards compatibility.

Returns:
1. success, boolean that indicates success
2. failure_message, string or nil
]]
function check.areTensorsEq(ta, tb, tolerance, negate, ignoreTensorDims)
   ignoreTensorDims = ignoreTensorDims or false

   if not ignoreTensorDims and ta:dim() ~= tb:dim() then
      return negate, 'The tensors have different dimensions'
   end

   if ta:type() ~= tb:type() then
      return negate, 'The tensors have different types'
   end

   -- If we are comparing two empty tensors, return true.
   -- This is needed because some functions below cannot be applied to tensors
   -- of dimension 0.
   if ta:dim() == 0 and tb:dim() == 0 then
      return not negate, 'Both tensors are empty'
   end

   local sameSize
   if ignoreTensorDims then
      sameSize = ta:nElement() == tb:nElement()
   else
      sameSize = ta:isSameSizeAs(tb)
   end
   if not sameSize then
      return negate, 'The tensors have different sizes'
   end

   return check.areSameFormatTensorsEq(ta, tb, tolerance, negate, false)
end

local typesMatching = {
      ['torch.ByteStorage'] = torch.ByteTensor,
      ['torch.CharStorage'] = torch.CharTensor,
      ['torch.ShortStorage'] = torch.ShortTensor,
      ['torch.IntStorage'] = torch.IntTensor,
      ['torch.LongStorage'] = torch.LongTensor,
      ['torch.FloatStorage'] = torch.FloatTensor,
      ['torch.DoubleStorage'] = torch.DoubleTensor,
      ['torch.HalfStorage'] = torch.HalfTensor,
}

--[[ Tests for storage equality.

Tests whether the maximum element-wise difference between `sa` and `sb` is less
than or equal to `tolerance`.

Arguments:
* `sa` (storage)
* `sb` (storage)
* `tolerance` (number) maximum elementwise difference between `a` and `b`.
* `negate` (boolean) if negate is true, we invert success and failure.

Returns:
1. success, boolean that indicates success
2. failure_message, string or nil
]]
function check.areStoragesEq(sa, sb, tolerance, negate)
   if sa:size() ~= sb:size() then
      return negate, 'The storages have different sizes'
   end

   local typeOfsa = torch.type(sa)
   local typeOfsb = torch.type(sb)

   if typeOfsa ~= typeOfsb then
      return negate, 'The storages have different types'
   end

   local ta = typesMatching[typeOfsa](sa)
   local tb = typesMatching[typeOfsb](sb)

   return check.areSameFormatTensorsEq(ta, tb, tolerance, negate, true)
end

--[[ Tests for general (deep) equality.

The types of `got` and `expected` must match.
Tables are compared recursively. Keys and types of the associated values must
match, recursively. Numbers are compared with the given tolerance.
Torch tensors and storages are compared with the given tolerance on their
elementwise difference. Other types are compared for strict equality with the
regular Lua == operator.

Arguments:
* `got`
* `expected`
* `tolerance` (number) maximum elementwise difference between `a` and `b`.
* `negate` (boolean) if negate is true, we invert success and failure.

Returns:
1. success, boolean that indicates success
2. failure_message, string or nil
]]
function check.areEq(got, expected, tolerance, negate)
   local errMessage
   if type(got) ~= type(expected) then
      if not negate then
         errMessage = 'EQ failed: values have different types (first: '
                      .. type(got) .. ', second: ' .. type(expected) .. ')'
      end
      return negate, errMessage
   elseif type(got) == 'number' then
      local diff = math.abs(got - expected)
      local ok = (diff <= tolerance)
      if negate then
         ok = not ok
      end
      if not ok then
         if negate then
            errMessage = string.format("NE failed: %s == %s",
                                       tostring(got), tostring(expected))
         else
            errMessage = string.format("EQ failed: %s ~= %s",
                                       tostring(got), tostring(expected))
         end
         if tolerance > 0 then
            errMessage = errMessage .. " with tolerance=" .. tostring(tolerance)
         end
      end
      return ok, errMessage
   elseif type(expected) == "table" then
     return check.areTablesEq(got, expected, tolerance, negate)
   elseif torch.isTensor(got) then
     return check.areTensorsEq(got, expected, tolerance, negate)
   elseif torch.isStorage(got) then
     return check.areStoragesEq(got, expected, tolerance, negate)
   else
     -- Below: we have the same type which is either userdata or a lua type
     -- which is not a number.
     local ok = (got == expected)
     if negate then
        ok = not ok
     end
     if not ok then
        if negate then
           errMessage = string.format("NE failed: %s (%s) == %s (%s)",
                                      tostring(got), type(got),
                                      tostring(expected), type(expected))
        else
           errMessage = string.format("EQ failed: %s (%s) ~= %s (%s)",
                                      tostring(got), type(got),
                                      tostring(expected), type(expected))
        end
     end
     return ok, errMessage
   end
end

--[[ Tests for (deep) table equality.

Tables are compared recursively. Keys and types of the associated values must
match, recursively. Numbers are compared with the given tolerance.
Torch tensors and storages are compared with the given tolerance on their
elementwise difference. Other types are compared for strict equality with the
regular Lua == operator.

Arguments:
* `t1` (table)
* `t2` (table)
* `tolerance` (number) maximum elementwise difference between `a` and `b`.
* `negate` (boolean) if negate is true, we invert success and failure.

Returns:
1. success, boolean that indicates success
2. failure_message, string or nil
]]
function check.areTablesEq(t1, t2, tolerance, negate)
   -- Implementation detail: Instead of doing a depth-first table comparison
   -- check (for example, using recursion), let's do a breadth-first search
   -- using a queue. Why? Because if we have two tables that are quite deep
   -- (e.g., a gModule from nngraph), then if they are different then it's
   -- more useful to the user to show how they differ at as-shallow-a-depth
   -- as possible.
   local queue = {}
   queue._head = 1
   queue._tail = 1
   function queue.isEmpty()
      return queue._tail == queue._head
   end
   function queue.pop()
      queue._head = queue._head + 1
      return queue[queue._head - 1]
   end
   function queue.push(value)
      queue[queue._tail] = value
      queue._tail = queue._tail + 1
   end

   queue.push({t1, t2})
   while not queue.isEmpty() do
      local location
      t1, t2, location = unpack(queue.pop())

      local function toSublocation(key)
         local keyAsString = tostring(key)
         return (location and location .. "." .. keyAsString) or keyAsString
      end

      for key, value1 in pairs(t1) do
         local sublocation = toSublocation(key)
         if t2[key] == nil then
            return negate, string.format(
                  "Entry %s missing in second table (is %s in first)",
                  sublocation, tostring(value1))
         end
         local value2 = t2[key]
         if type(value1) == 'table' and type(value2) == 'table' then
            queue.push({value1, value2, sublocation})
         else
            local ok, message = check.areEq(value1, value2, tolerance, false)
            if not ok then
               message = 'At table location ' .. sublocation .. ': ' .. message
               return negate, message
            end
         end
      end

      for key, value2 in pairs(t2) do
         local sublocation = toSublocation(key)
         if t1[key] == nil then
             return negate, string.format(
                   "Entry %s missing in first table (is %s in second)",
                   sublocation, tostring(value2))
         end
      end
   end
   return not negate, 'The tables are equal'
end
