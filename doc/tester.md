<a name="torch.Tester.dok"></a>
# Tester #

This class provides a generic unit testing framework. It is already 
being used in [nn](../index.md) package to verify the correctness of classes.

The framework is generally used as follows.

```lua
mytest = {}

tester = torch.Tester()

function mytest.TestA()
	local a = 10
	local b = 10
	tester:asserteq(a,b,'a == b')
	tester:assertne(a,b,'a ~= b')
end

function mytest.TestB()
	local a = 10
	local b = 9
	tester:assertlt(a,b,'a < b')
	tester:assertgt(a,b,'a > b')
end

tester:add(mytest)
tester:run()

```

Running this code will report 2 errors in 2 test functions. Generally it is 
better to put a single test case in each test function unless several very related
test cases exist. The error report includes the message and line number of the error.

```

Running 2 tests
**  ==> Done 

Completed 2 tests with 2 errors

--------------------------------------------------------------------------------
TestB
a < b
 LT(<) violation   val=10, condition=9
	...y/usr.t7/local.master/share/lua/5.1/torch/Tester.lua:23: in function 'assertlt'
	[string "function mytest.TestB()..."]:4: in function 'f'

--------------------------------------------------------------------------------
TestA
a ~= b
 NE(~=) violation   val=10, condition=10
	...y/usr.t7/local.master/share/lua/5.1/torch/Tester.lua:38: in function 'assertne'
	[string "function mytest.TestA()..."]:5: in function 'f'

--------------------------------------------------------------------------------

```


<a name="torch.Tester"></a>
### torch.Tester() ###

Returns a new instance of `torch.Tester` class.

<a name="torch.Tester.add"></a>
### add(f, 'name') ###

Adds a new test function with name `name`. The test function is stored in `f`.
The function is supposed to run without any arguments and not return any values.

<a name="torch.Tester.add"></a>
### add(ftable) ###

Recursively adds all function entries of the table `ftable` as tests. This table 
can only have functions or nested tables of functions.

<a name="torch.Tester.assert"></a>
### assert(condition [, message]) ###

Saves an error if condition is not true with the optional message.

<a name="torch.Tester.assertlt"></a>
### assertlt(val, condition [, message]) ###

Saves an error if `val < condition` is not true with the optional message.

<a name="torch.Tester.assertgt"></a>
### assertgt(val, condition [, message]) ###

Saves an error if `val > condition` is not true with the optional message.

<a name="torch.Tester.assertle"></a>
### assertle(val, condition [, message]) ###

Saves an error if `val <= condition` is not true with the optional message.

<a name="torch.Tester.assertge"></a>
### assertge(val, condition [, message]) ###

Saves an error if `val >= condition` is not true with the optional message.

<a name="torch.Tester.asserteq"></a>
### asserteq(val, condition [, message]) ###

Saves an error if `val == condition` is not true with the optional message.

<a name="torch.Tester.assertne"></a>
### assertne(val, condition [, message]) ###

Saves an error if `val ~= condition` is not true with the optional message.

<a name="torch.Tester.assertTensorEq"></a>
### assertTensorEq(ta, tb, condition [, message]) ###

Saves an error if `max(abs(ta-tb)) < condition` is not true with the optional message.

<a name="torch.Tester.assertTensorNe"></a>
### assertTensorNe(ta, tb, condition [, message]) ###

Saves an error if `max(abs(ta-tb)) >= condition` is not true with the optional message.

<a name="torch.Tester.assertTableEq"></a>
### assertTableEq(ta, tb, condition [, message]) ###

Saves an error if `max(abs(ta-tb)) < condition` is not true with the optional message.

<a name="torch.Tester.assertTableNe"></a>
### assertTableNe(ta, tb, condition [, message]) ###

Saves an error if `max(abs(ta-tb)) >= condition` is not true with the optional message.

<a name="torch.Tester.assertError"></a>
### assertError(f [, message]) ###

Saves an error if calling the function f() does not return an error, with the optional message.

<a name="torch.Tester.run"></a>
### run() ###

Runs all the test functions that are stored using [add()](#torch.Tester.add) function. 
While running it reports progress and at the end gives a summary of all errors.







