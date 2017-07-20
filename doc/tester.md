<a name="torch.Tester.dok"></a>
# Tester #

This class provides a generic unit testing framework. It is already
being used in [nn](../index.md) package to verify the correctness of classes.

The framework is generally used as follows.

```lua
local mytest = torch.TestSuite()

local tester = torch.Tester()

function mytest.testA()
   local a = torch.Tensor{1, 2, 3}
   local b = torch.Tensor{1, 2, 4}
   tester:eq(a, b, "a and b should be equal")
end

function mytest.testB()
   local a = {2, torch.Tensor{1, 2, 2}}
   local b = {2, torch.Tensor{1, 2, 2.001}}
   tester:eq(a, b, 0.01, "a and b should be approximately equal")
end

function mytest.testC()
   local function myfunc()
      return "hello " .. world
   end
   tester:assertNoError(myfunc, "myfunc shouldn't give an error")
end

tester:add(mytest)
tester:run()
```

Running this code will report two test failures (and one test success).
Generally it is  better to put a single test case in each test function unless
several very related test cases exist.
The error report includes the message and line number of the error.

```
Running 3 tests
1/3 testB ............................................................... [PASS]
2/3 testA ............................................................... [FAIL]
3/3 testC ............................................................... [FAIL]
Completed 3 asserts in 3 tests with 2 failures and 0 errors
--------------------------------------------------------------------------------
testA
a and b should be equal
TensorEQ(==) violation: max diff=1, tolerance=0
stack traceback:
        ./test.lua:8: in function <./test.lua:5>

--------------------------------------------------------------------------------
testC
myfunc shouldn't give an error
ERROR violation: err=./test.lua:19: attempt to concatenate global 'world' (a nil value)
stack traceback:
        ./test.lua:21: in function <./test.lua:17>

--------------------------------------------------------------------------------
torch/torch/Tester.lua:383: An error was found while running tests!
stack traceback:
        [C]: in function 'assert'
        torch/torch/Tester.lua:383: in function 'run'
        ./test.lua:25: in main chunk
```

Historically, Tester has supported a variety of equality checks
([asserteq](#torch.Tester.asserteq),
[assertalmosteq](#torch.Tester.assertalmosteq),
[assertTensorEq](#torch.Tester.assertTensorEq),
[assertTableEq](#torch.Tester.assertTableEq), and their negations). In general
however, you should just use [eq](#torch.Tester.eq) (or its negation
[ne](#torch.Tester.ne)).  These functions do deep checking of many object types
including recursive tables and tensors, and support a
tolerance parameter for comparing numerical values (including tensors).

Many of the tester functions accept both an optional `tolerance` parameter and a
`message` to display if the test case fails. For both convenience and backwards
compatibility, these arguments can be supplied in either order.

<a name="torch.Tester"></a>
### torch.Tester() ###

Returns a new instance of `torch.Tester` class.

<a name="torch.Tester.add"></a>
### add(f, 'name') ###

Adds `f`, either a test function or a table of test functions, to the tester.

If `f` is a function then names should be unique. There are a couple of special
values for `name`: if it is `_setUp` or `_tearDown`, then the function will be
called either *before* or *after* every test respectively, with the name of the
test passed as a parameter.

If `f` is a table then `name` should be nil, and the names of the individual
tests in the table will be taken from the corresponding table key. It's
recommended you use [TestSuite](#torch.TestSuite.dok) for tables of tests.

Returns the torch.Tester instance.

<a name="torch.Tester.run"></a>
### run(testNames) ###

Runs tests that have been added by [add(f, 'name')](#torch.Tester.add).
While running it reports progress, and at the end gives a summary of all errors.

If a list of names `testNames` is passed, then all tests matching these names
(using `string.match`) will be run; otherwise all tests will be run.

```lua
tester:run() -- runs all tests
tester:run("test1") -- runs the test named "test1"
tester:run({"test2", "test3"}) -- runs the tests named "test2" and "test3"
```

<a name="torch.Tester.disable"></a>
### disable(testNames) ###

Prevents the given tests from running, where `testNames` can be a single string
or list of strings. More precisely, when [run](#torch.Tester.run)
is invoked, it will skip these tests, while still printing out an indication of
skipped tests. This is useful for temporarily disabling tests without
commenting out the code (for example, if they depend on upstream code that is
currently broken), and explicitly flagging them as skipped.

Returns the torch.Tester instance.

```lua
local tester = torch.Tester()
local tests = torch.TestSuite()

function tests.brokenTest()
  -- ...
end

tester:add(tests):disable('brokenTest'):run()
```

```
Running 1 test
1/1 brokenTest .......................................................... [SKIP]
Completed 0 asserts in 1 test with 0 failures and 0 errors and 1 disabled
```

<a name="torch.Tester.assert"></a>
### assert(condition [, message]) ###

Checks that `condition` is true (using the optional `message` if the test
fails).
Returns whether the test passed.

<a name="torch.Tester.assertGeneralEq"></a>
### assertGeneralEq(got, expected [, tolerance] [, message]) ###

General equality check between numbers, tables, strings, `torch.Tensor`
objects, `torch.Storage` objects, etc.

Checks that `got` and `expected` have the same contents, where tables are
compared recursively, tensors and storages are compared elementwise, and numbers
are compared within `tolerance` (default value `0`). Other types are compared by
strict equality. The optional `message` is used if the test fails.
Returns whether the test passed.

<a name="torch.Tester.eq"></a>
### eq(got, expected  [, tolerance] [, message]) ###

Convenience function; does the same as
[assertGeneralEq](#torch.Tester.assertGeneralEq).

<a name="torch.Tester.assertGeneralNe"></a>
### assertGeneralNe(got, unexpected  [, tolerance] [, message]) ###

General inequality check between numbers, tables, strings, `torch.Tensor`
objects, `torch.Storage` objects, etc.

Checks that `got` and `unexpected` have different contents, where tables are
compared recursively, tensors and storages are compared elementwise, and numbers
are compared within `tolerance` (default value `0`). Other types are compared by
strict equality. The optional `message` is used if the test fails.
Returns whether the test passed.

<a name="torch.Tester.ne"></a>
### ne(got, unexpected  [, tolerance] [, message]) ###

Convenience function; does the same as
[assertGeneralNe](#torch.Tester.assertGeneralNe).

<a name="torch.Tester.assertlt"></a>
### assertlt(a, b [, message]) ###

Checks that `a < b` (using the optional `message` if the test fails),
where `a` and `b` are numbers.
Returns whether the test passed.

<a name="torch.Tester.assertgt"></a>
### assertgt(a, b [, message]) ###

Checks that `a > b` (using the optional `message` if the test fails),
where `a` and `b` are numbers.
Returns whether the test passed.

<a name="torch.Tester.assertle"></a>
### assertle(a, b [, message]) ###

Checks that `a <= b` (using the optional `message` if the test fails),
where `a` and `b` are numbers.
Returns whether the test passed.

<a name="torch.Tester.assertge"></a>
### assertge(a, b [, message]) ###

Checks that `a >= b` (using the optional `message` if the test fails),
where `a` and `b` are numbers.
Returns whether the test passed.

<a name="torch.Tester.asserteq"></a>
### asserteq(a, b [, message]) ###

Checks that `a == b` (using the optional `message` if the test fails).
Note that this uses the generic lua equality check, so objects such as tensors
that have the same content but are distinct objects will fail this test;
consider using [assertGeneralEq()](#torch.Tester.assertGeneralEq) instead.
Returns whether the test passed.

<a name="torch.Tester.assertne"></a>
### assertne(a, b [, message]) ###

Checks that `a ~= b` (using the optional `message` if the test fails).
Note that this uses the generic lua inequality check, so objects such as tensors
that have the same content but are distinct objects will pass this test;
consider using [assertGeneralNe()](#torch.Tester.assertGeneralNe) instead.
Returns whether the test passed.

<a name="torch.Tester.assertalmosteq"></a>
### assertalmosteq(a, b [, tolerance] [, message]) ###

Checks that `|a - b| <= tolerance` (using the optional `message` if the
test fails), where `a` and `b` are numbers, and `tolerance` is an optional
number (default `1e-16`).
Returns whether the test passed.

<a name="torch.Tester.assertTensorEq"></a>
### assertTensorEq(ta, tb [, tolerance] [, message]) ###

Checks that `max(abs(ta - tb)) <= tolerance` (using the optional `message`
if the test fails), where `ta` and `tb` are tensors, and `tolerance` is an
optional number (default `1e-16`). Tensors that are different types or sizes
will cause this check to fail.
Returns whether the test passed.

<a name="torch.Tester.assertTensorNe"></a>
### assertTensorNe(ta, tb [, tolerance] [, message]) ###

Checks that `max(abs(ta - tb)) > tolerance` (using the optional `message`
if the test fails), where `ta` and `tb` are tensors, and `tolerance` is an
optional number (default `1e-16`). Tensors that are different types or sizes
will cause this check to pass.
Returns whether the test passed.

<a name="torch.Tester.assertTableEq"></a>
### assertTableEq(ta, tb [, tolerance] [, message]) ###

Checks that the two tables have the same contents, comparing them
recursively, where objects such as tensors are compared using their contents.
Numbers (such as those appearing in tensors) are considered equal if
their difference is at most the given tolerance.

<a name="torch.Tester.assertTableNe"></a>
### assertTableNe(ta, tb [, tolerance] [, message]) ###

Checks that the two tables have distinct contents, comparing them
recursively, where objects such as tensors are compared using their contents.
Numbers (such as those appearing in tensors) are considered equal if
their difference is at most the given tolerance.

<a name="torch.Tester.assertError"></a>
### assertError(f [, message]) ###

Checks that calling `f()` (via `pcall`) raises an error (using the
optional `message` if the test fails).
Returns whether the test passed.

<a name="torch.Tester.assertNoError"></a>
### assertNoError(f [, message]) ###

Check that calling `f()` (via `pcall`) does not raise an error (using the
optional `message` if the test fails).
Returns whether the test passed.

<a name="torch.Tester.assertErrorMsg"></a>
### assertErrorMsg(f, errmsg [, message]) ###

Checks that calling `f()` (via `pcall`) raises an error with the specific error
message `errmsg` (using the optional `message` if the test fails).
Returns whether the test passed.

<a name="torch.Tester.assertErrorPattern"></a>
### assertErrorPattern(f, errPattern [, message]) ###

Checks that calling `f()` (via `pcall`) raises an error matching `errPattern`
(using the optional `message` if the test fails).
The matching is done using `string.find`; in particular substrings will match.
Returns whether the test passed.

<a name="torch.Tester.assertErrorObj"></a>
### assertErrorObj(f, errcomp [, message]) ###

Checks that calling `f()` (via `pcall`) raises an error object `err` such that
calling `errcomp(err)` returns true (using the optional `message` if the test
fails).
Returns whether the test passed.

<a name="torch.Tester.setEarlyAbort"></a>
### setEarlyAbort(earlyAbort) ###

If `earlyAbort == true` then the testing will stop on the first test failure.
By default this is off.

<a name="torch.Tester.setRethrowErrors"></a>
### setRethrowErrors(rethrowErrors) ###

If `rethrowErrors == true` then lua errors encountered during the execution of
the tests will be rethrown, instead of being caught by the tester.
By default this is off.

<a name="torch.Tester.setSummaryOnly"></a>
### setSummaryOnly(summaryOnly) ###

If `summaryOnly == true`, then only the pass / fail status of the tests will be
printed out, rather than full error messages. By default, this is off.


<a name="torch.TestSuite.dok"></a>
# TestSuite #

A TestSuite is used in conjunction with [Tester](#torch.Tester.dok). It is
created via `torch.TestSuite()`, and behaves like a plain lua table,
except that it also checks that duplicate tests are not created.
It is recommended that you always use a TestSuite instead of a plain table for
your tests.

The following example code attempts to add a function with the same name
twice to a TestSuite (a surprisingly common mistake), which gives an error.

```lua
> test = torch.TestSuite()
>
> function test.myTest()
>    -- ...
> end
>
> -- ...
>
> function test.myTest()
>    -- ...
> end
torch/TestSuite.lua:16: Test myTest is already defined.
```

