require 'torch'

local tester = torch.Tester()

local MESSAGE = "a really useful informative error message"

local subtester = torch.Tester()
-- The message only interests us in case of failure
subtester._success = function(self) return true, MESSAGE end
subtester._failure = function(self, message) return false, message end

local tests = torch.TestSuite()

local test_name_passed_to_setUp
local calls_to_setUp = 0
local calls_to_tearDown = 0

local originalIoWrite = io.write
local function disableIoWrite()
   io.write = function() end
end
local function enableIoWrite()
   io.write = originalIoWrite
end

local function meta_assert_success(success, message)
   tester:assert(success == true, "assert wasn't successful")
   tester:assert(string.find(message, MESSAGE) ~= nil, "message doesn't match")
end
local function meta_assert_failure(success, message)
   tester:assert(success == false, "assert didn't fail")
   tester:assert(string.find(message, MESSAGE) ~= nil, "message doesn't match")
end

function tests.really_test_assert()
   assert((subtester:assert(true, MESSAGE)),
          "subtester:assert doesn't actually work!")
   assert(not (subtester:assert(false, MESSAGE)),
          "subtester:assert doesn't actually work!")
end

function tests.setEarlyAbort()
   disableIoWrite()

   for _, earlyAbort in ipairs{false, true} do
      local myTester = torch.Tester()

      local invokedCount = 0
      local myTests = {}
      function myTests.t1()
         invokedCount = invokedCount + 1
         myTester:assert(false)
      end
      myTests.t2 = myTests.t1

      myTester:setEarlyAbort(earlyAbort)
      myTester:add(myTests)
      pcall(myTester.run, myTester)

      tester:assert(invokedCount == (earlyAbort and 1 or 2),
                    "wrong number of tests invoked for use with earlyAbort")
   end

   enableIoWrite()
end

function tests.setRethrowErrors()
   disableIoWrite()

   local myTester = torch.Tester()
   myTester:setRethrowErrors(true)
   myTester:add(function() error("a throw") end)

   tester:assertErrorPattern(function() myTester:run() end,
                             "a throw",
                             "error should be rethrown")

   enableIoWrite()
end

function tests.disable()
   disableIoWrite()

   for disableCount = 1, 2 do
      local myTester = torch.Tester()
      local tests = {}
      local test1Invoked = false
      local test2Invoked = false
      function tests.test1()
         test1Invoked = true
      end
      function tests.test2()
         test2Invoked = true
      end
      myTester:add(tests)

      if disableCount == 1 then
         myTester:disable('test1'):run()
         tester:assert((not test1Invoked) and test2Invoked,
                       "disabled test shouldn't have been invoked")
      else
         myTester:disable({'test1', 'test2'}):run()
         tester:assert((not test1Invoked) and (not test2Invoked),
                       "disabled tests shouldn't have been invoked")
      end
   end

   enableIoWrite()
end

function tests.assert()
   meta_assert_success(subtester:assert(true, MESSAGE))
   meta_assert_failure(subtester:assert(false, MESSAGE))
end

local function testEqNe(eqExpected, ...)
   if eqExpected then
      meta_assert_success(subtester:eq(...))
      meta_assert_failure(subtester:ne(...))
   else
      meta_assert_failure(subtester:eq(...))
      meta_assert_success(subtester:ne(...))
   end
end

--[[ Test :assertGeneralEq and :assertGeneralNe (also known as :eq and :ne).

Note that in-depth testing of testing of many specific types of data (such as
Tensor) is covered below, when we test specific functions (such as
:assertTensorEq). This just does a general check, as well as testing of testing
of mixed datatypes.
]]
function tests.assertGeneral()
   local one = torch.Tensor{1}

   testEqNe(true, one, one, MESSAGE)
   testEqNe(false, one, 1, MESSAGE)
   testEqNe(true, "hi", "hi", MESSAGE)
   testEqNe(true, {one, 1}, {one, 1}, MESSAGE)
   testEqNe(true, {{{one}}}, {{{one}}}, MESSAGE)
   testEqNe(false, {{{one}}}, {{one}}, MESSAGE)
   testEqNe(true, torch.Storage{1}, torch.Storage{1}, MESSAGE)
   testEqNe(false, torch.FloatStorage{1}, torch.LongStorage{1}, MESSAGE)
   testEqNe(false, torch.Storage{1}, torch.Storage{1, 2}, MESSAGE)
   testEqNe(false, "one", 1, MESSAGE)
   testEqNe(false, {one}, {one + torch.Tensor{1e-10}}, MESSAGE)
   testEqNe(true, {one}, {one + torch.Tensor{1e-10}}, 1e-9, MESSAGE)
end

function tests.assertlt()
   meta_assert_success(subtester:assertlt(1, 2, MESSAGE))
   meta_assert_failure(subtester:assertlt(2, 1, MESSAGE))
   meta_assert_failure(subtester:assertlt(1, 1, MESSAGE))
end

function tests.assertgt()
   meta_assert_success(subtester:assertgt(2, 1, MESSAGE))
   meta_assert_failure(subtester:assertgt(1, 2, MESSAGE))
   meta_assert_failure(subtester:assertgt(1, 1, MESSAGE))
end

function tests.assertle()
   meta_assert_success(subtester:assertle(1, 2, MESSAGE))
   meta_assert_failure(subtester:assertle(2, 1, MESSAGE))
   meta_assert_success(subtester:assertle(1, 1, MESSAGE))
end

function tests.assertge()
   meta_assert_success(subtester:assertge(2, 1, MESSAGE))
   meta_assert_failure(subtester:assertge(1, 2, MESSAGE))
   meta_assert_success(subtester:assertge(1, 1, MESSAGE))
end

function tests.asserteq()
   meta_assert_success(subtester:asserteq(1, 1, MESSAGE))
   meta_assert_failure(subtester:asserteq(1, 2, MESSAGE))
end

function tests.assertalmosteq()
   meta_assert_success(subtester:assertalmosteq(1, 1, MESSAGE))
   meta_assert_success(subtester:assertalmosteq(1, 1 + 1e-17, MESSAGE))
   meta_assert_success(subtester:assertalmosteq(1, 2, 2, MESSAGE))
   meta_assert_failure(subtester:assertalmosteq(1, 2, MESSAGE))
   meta_assert_failure(subtester:assertalmosteq(1, 3, 1, MESSAGE))
end

function tests.assertne()
   meta_assert_success(subtester:assertne(1, 2, MESSAGE))
   meta_assert_failure(subtester:assertne(1, 1, MESSAGE))
end

-- The `alsoTestEq` flag is provided to test :eq in addition to :assertTensorEq.
-- The behaviour of the two isn't always the same due to handling of tensors of
-- different dimensions but the same number of elements.
local function testTensorEqNe(eqExpected, alsoTestEq, ...)
   if eqExpected then
      meta_assert_success(subtester:assertTensorEq(...))
      meta_assert_failure(subtester:assertTensorNe(...))
      if alsoTestEq then
         meta_assert_success(subtester:eq(...))
         meta_assert_failure(subtester:ne(...))
      end
   else
      meta_assert_failure(subtester:assertTensorEq(...))
      meta_assert_success(subtester:assertTensorNe(...))
      if alsoTestEq then
         meta_assert_failure(subtester:eq(...))
         meta_assert_success(subtester:ne(...))
      end
   end
end

function tests.assertTensor_types()
   local allTypes = {
         torch.ByteTensor,
         torch.CharTensor,
         torch.ShortTensor,
         torch.IntTensor,
         torch.LongTensor,
         torch.FloatTensor,
         torch.DoubleTensor,
   }
   for _, tensor1 in ipairs(allTypes) do
      for _, tensor2 in ipairs(allTypes) do
         local t1 = tensor1():ones(10)
         local t2 = tensor2():ones(10)
         testTensorEqNe(tensor1 == tensor2, true, t1, t2, 1e-6, MESSAGE)
      end
   end

   testTensorEqNe(false, true, torch.FloatTensor(), torch.LongTensor(), MESSAGE)
end

function tests.assertTensor_sizes()
   local t = torch.Tensor() -- no dimensions
   local t2 = torch.ones(2)
   local t3 = torch.ones(3)
   local t12 = torch.ones(1, 2)
   assert(subtester._assertTensorEqIgnoresDims == true) -- default state
   testTensorEqNe(false, false, t, t2, 1e-6, MESSAGE)
   testTensorEqNe(false, false, t, t3, 1e-6, MESSAGE)
   testTensorEqNe(false, false, t, t12, 1e-6, MESSAGE)
   testTensorEqNe(false, false, t2, t3, 1e-6, MESSAGE)
   testTensorEqNe(true, false, t2, t12, 1e-6, MESSAGE)
   testTensorEqNe(false, false, t3, t12, 1e-6, MESSAGE)
   subtester._assertTensorEqIgnoresDims = false
   testTensorEqNe(false, true, t, t2, 1e-6, MESSAGE)
   testTensorEqNe(false, true, t, t3, 1e-6, MESSAGE)
   testTensorEqNe(false, true, t, t12, 1e-6, MESSAGE)
   testTensorEqNe(false, true, t2, t3, 1e-6, MESSAGE)
   testTensorEqNe(false, true, t2, t12, 1e-6, MESSAGE)
   testTensorEqNe(false, true, t3, t12, 1e-6, MESSAGE)
   subtester._assertTensorEqIgnoresDims = true -- reset back
end

function tests.assertTensor_epsilon()
   local t1 = torch.rand(100, 100)
   local t2 = torch.rand(100, 100) * 1e-5
   local t3 = t1 + t2
   testTensorEqNe(true, true, t1, t3, 1e-4, MESSAGE)
   testTensorEqNe(false, true, t1, t3, 1e-6, MESSAGE)
end

function tests.assertTensor_arg()
   local one = torch.Tensor{1}

   tester:assertErrorPattern(
         function() subtester:assertTensorEq(one, 2) end,
         "Second argument should be a Tensor")

   -- Test that assertTensorEq support message and tolerance in either ordering
   tester:assertNoError(
         function() subtester:assertTensorEq(one, one, 0.1, MESSAGE) end)
   tester:assertNoError(
         function() subtester:assertTensorEq(one, one, MESSAGE, 0.1) end)
end

function tests.assertTensor()
   local t1 = torch.randn(100, 100)
   local t2 = t1:clone()
   local t3 = torch.randn(100, 100)
   testTensorEqNe(true, true, t1, t2, 1e-6, MESSAGE)
   testTensorEqNe(false, true, t1, t3, 1e-6, MESSAGE)
   testTensorEqNe(true, true, torch.Tensor(), torch.Tensor(), MESSAGE)
end

-- Check that calling assertTensorEq with two tensors with the same content but
-- different dimensions gives a warning.
function tests.assertTensorDimWarning()
   local myTester = torch.Tester()
   myTester:add(
       function()
          myTester:assertTensorEq(torch.Tensor{{1}}, torch.Tensor{1})
       end)

   local warningGiven = false
   io.write = function(s)
      if string.match(s, 'but different dimensions') then
         warningGiven = true
      end
   end

   myTester:run()
   enableIoWrite()

   tester:assert(warningGiven,
                 "Calling :assertTensorEq({{1}}, {1}) should give a warning")
end

local function testTableEqNe(eqExpected, ...)
   if eqExpected then
      meta_assert_success(subtester:assertTableEq(...))
      meta_assert_failure(subtester:assertTableNe(...))
      meta_assert_success(subtester:eq(...))
      meta_assert_failure(subtester:ne(...))
   else
      meta_assert_failure(subtester:assertTableEq(...))
      meta_assert_success(subtester:assertTableNe(...))
      meta_assert_failure(subtester:eq(...))
      meta_assert_success(subtester:ne(...))
   end
end

function tests.assertTable()
   testTableEqNe(true, {1, 2, 3}, {1, 2, 3}, MESSAGE)
   testTableEqNe(false, {1, 2, 3}, {3, 2, 1}, MESSAGE)
   testTableEqNe(true, {1, 2, {4, 5}}, {1, 2, {4, 5}}, MESSAGE)
   testTableEqNe(false, {1, 2, 3}, {1,2}, MESSAGE)
   testTableEqNe(false, {1, 2, 3}, {1, 2, 3, 4}, MESSAGE)
   testTableEqNe(true, {{1}}, {{1}}, MESSAGE)
   testTableEqNe(false, {{1}}, {{{1}}}, MESSAGE)
   testTableEqNe(true, {false}, {false}, MESSAGE)
   testTableEqNe(false, {true}, {false}, MESSAGE)
   testTableEqNe(false, {false}, {true}, MESSAGE)

   local tensor = torch.rand(100, 100)
   local t1 = {1, "a", key = "value", tensor = tensor, subtable = {"nested"}}
   local t2 = {1, "a", key = "value", tensor = tensor, subtable = {"nested"}}
   testTableEqNe(true, t1, t2, MESSAGE)
   for k, v in pairs(t1) do
      local x = "something else"
      t2[k] = nil
      t2[x] = v
      testTableEqNe(false, t1, t2, MESSAGE)
      t2[x] = nil
      t2[k] = x
      testTableEqNe(false, t1, t2, MESSAGE)
      t2[k] = v
      testTableEqNe(true, t1, t2, MESSAGE)
   end
end

local function good_fn() end
local function bad_fn() error("muahaha!") end

function tests.assertError()
   meta_assert_success(subtester:assertError(bad_fn, MESSAGE))
   meta_assert_failure(subtester:assertError(good_fn, MESSAGE))
end

function tests.assertNoError()
   meta_assert_success(subtester:assertNoError(good_fn, MESSAGE))
   meta_assert_failure(subtester:assertNoError(bad_fn, MESSAGE))
end

function tests.assertErrorPattern()
   meta_assert_success(subtester:assertErrorPattern(bad_fn, "haha", MESSAGE))
   meta_assert_failure(subtester:assertErrorPattern(bad_fn, "hehe", MESSAGE))
end

function tests.testSuite_duplicateTests()
   local function createDuplicateTests()
      local tests = torch.TestSuite()
      function tests.testThis() end
      function tests.testThis() end
   end
   tester:assertErrorPattern(createDuplicateTests,
                             "Test testThis is already defined.")
end

--[[ Returns a Tester with `numSuccess` success cases, `numFailure` failure
  cases, and with an error if `hasError` is true.
  Success and fail tests are evaluated with tester:eq
]]
local function genDummyTest(numSuccess, numFailure, hasError)
   hasError = hasError or false

   local dummyTester = torch.Tester()
   local dummyTests = torch.TestSuite()

   if numSuccess > 0 then
      function dummyTests.testDummySuccess()
         for i = 1, numSuccess do
           dummyTester:eq({1}, {1}, '', 0)
         end
      end
   end

   if numFailure > 0 then
      function dummyTests.testDummyFailure()
         for i = 1, numFailure do
            dummyTester:eq({1}, {2}, '', 0)
         end
      end
   end

   if hasError then
      function dummyTests.testDummyError()
         error('dummy error')
      end
   end

   return dummyTester:add(dummyTests)
end

function tests.runStatusAndAssertCounts()
   local emptyTest      = genDummyTest(0, 0, false)
   local sucTest        = genDummyTest(1, 0, false)
   local multSucTest    = genDummyTest(4, 0, false)
   local failTest       = genDummyTest(0, 1, false)
   local errTest        = genDummyTest(0, 0, true)
   local errFailTest    = genDummyTest(0, 1, true)
   local errSucTest     = genDummyTest(1, 0, true)
   local failSucTest    = genDummyTest(1, 1, false)
   local failSucErrTest = genDummyTest(1, 1, true)

   disableIoWrite()

   local success, msg = pcall(emptyTest.run, emptyTest)
   tester:asserteq(success, true, "pcall should succeed for empty tests")

   local success, msg = pcall(sucTest.run, sucTest)
   tester:asserteq(success, true, "pcall should succeed for 1 successful test")

   local success, msg = pcall(multSucTest.run, multSucTest)
   tester:asserteq(success, true,
                   "pcall should succeed for 2+ successful tests")

   local success, msg = pcall(failTest.run, failTest)
   tester:asserteq(success, false, "pcall should fail for tests with failure")

   local success, msg = pcall(errTest.run, errTest)
   tester:asserteq(success, false, "pcall should fail for tests with error")

   local success, msg = pcall(errFailTest.run, errFailTest)
   tester:asserteq(success, false, "pcall should fail for error+fail tests")

   local success, msg = pcall(errSucTest.run, errSucTest)
   tester:asserteq(success, false, "pcall should fail for error+success tests")

   local success, msg = pcall(failSucTest.run, failSucTest)
   tester:asserteq(success, false, "pcall should fail for fail+success tests")

   local success, msg = pcall(failSucErrTest.run, failSucErrTest)
   tester:asserteq(success, false,
                   "pcall should fail for fail+success+err test")

   enableIoWrite()

   tester:asserteq(emptyTest.countasserts, 0,
                   "emptyTest should have 0 asserts")
   tester:asserteq(sucTest.countasserts, 1, "sucTest should have 1 assert")
   tester:asserteq(multSucTest.countasserts, 4,
                   "multSucTest should have 4 asserts")
   tester:asserteq(failTest.countasserts, 1, "failTest should have 1 assert")
   tester:asserteq(errTest.countasserts, 0, "errTest should have 0 asserts")
   tester:asserteq(errFailTest.countasserts, 1,
                   "errFailTest should have 1 assert")
   tester:asserteq(errSucTest.countasserts, 1,
                   "errSucTest should have 0 asserts")
   tester:asserteq(failSucTest.countasserts, 2,
                   "failSucTest should have 2 asserts")
end

function tests.checkNestedTestsForbidden()
   disableIoWrite()

   local myTester = torch.Tester()
   local myTests = {{function() end}}
   tester:assertErrorPattern(function() myTester:add(myTests) end,
                             "Nested sets",
                             "tester should forbid adding nested test sets")

   enableIoWrite()
end

function tests.checkWarningOnAssertObject()
   -- This test checks that calling assert with an object generates a warning
   local myTester = torch.Tester()
   local myTests = {}
   function myTests.assertAbuse()
      myTester:assert({})
   end
   myTester:add(myTests)

   local warningGiven = false
   io.write = function(s)
      if string.match(s, 'should only be used for boolean') then
         warningGiven = true
      end
   end

   myTester:run()
   enableIoWrite()

   tester:assert(warningGiven, "Should warn on calling :assert(object)")
end

function tests.checkWarningOnAssertNeObject()
   -- This test checks that calling assertne with two objects generates warning
   local myTester = torch.Tester()
   local myTests = {}
   function myTests.assertAbuse()
      myTester:assertne({}, {})
   end
   myTester:add(myTests)

   local warningGiven = false
   io.write = function(s)
      if string.match(s, 'assertne should only be used to compare basic') then
         warningGiven = true
      end
   end

   myTester:run()
   enableIoWrite()

   tester:assert(warningGiven, "Should warn on calling :assertne(obj, obj)")
end

function tests.checkWarningOnExtraAssertArguments()
   -- This test checks that calling assert with extra args gives a lua error
   local myTester = torch.Tester()
   local myTests = {}
   function myTests.assertAbuse()
      myTester:assert(true, "some message", "extra argument")
   end
   myTester:add(myTests)

   local errorGiven = false
   io.write = function(s)
      if string.match(s, 'Unexpected arguments') then
         errorGiven = true
      end
   end
   tester:assertError(function() myTester:run() end)
   enableIoWrite()

   tester:assert(errorGiven, ":assert should fail on extra arguments")
end

function tests.checkWarningOnUsingTable()
   -- Checks that if we don't use a TestSuite then gives a warning
   local myTester = torch.Tester()
   local myTests = {}
   myTester:add(myTests)

   local errorGiven = false
   io.write = function(s)
      if string.match(s, 'use TestSuite rather than plain lua table') then
         errorGiven = true
      end
   end
   myTester:run()

   enableIoWrite()
   tester:assert(errorGiven, "Using a plain lua table for testsuite should warn")
end

function tests.checkMaxAllowedSetUpAndTearDown()
   -- Checks can have at most 1 set-up and at most 1 tear-down function
   local function f() end
   local myTester = torch.Tester()

   for _, name in ipairs({'_setUp', '_tearDown'}) do
      tester:assertNoError(function() myTester:add(f, name) end,
                           "Adding 1 set-up / tear-down should be fine")
      tester:assertErrorPattern(function() myTester:add(f, name) end,
                                "Only one",
                                "Adding second set-up / tear-down should fail")
   end
end

function tests.test_setUp()
   tester:asserteq(test_name_passed_to_setUp, 'test_setUp')
   for key, value in pairs(tester.tests) do
      tester:assertne(key, '_setUp')
   end
end

function tests.test_tearDown()
   for key, value in pairs(tester.tests) do
      tester:assertne(key, '_tearDown')
   end
end

function tests._setUp(name)
   test_name_passed_to_setUp = name
   calls_to_setUp = calls_to_setUp + 1
end

function tests._tearDown(name)
   calls_to_tearDown = calls_to_tearDown + 1
end

tester:add(tests):run()

-- Additional tests to check that _setUp and _tearDown were called.
local test_count = 0
for _ in pairs(tester.tests) do
   test_count = test_count + 1
end
local postTests = torch.TestSuite()
local postTester = torch.Tester()

function postTests.test_setUp(tester)
   postTester:asserteq(calls_to_setUp, test_count,
                       "Expected " .. test_count .. " calls to _setUp")
end

function postTests.test_tearDown()
   postTester:asserteq(calls_to_tearDown, test_count,
                      "Expected " .. test_count .. " calls to _tearDown")
end

postTester:add(postTests):run()
