<a name="torch.maths.dok"/>
# Math Functions #

Torch provides Matlab-like functions for manipulating
[Tensor](README.md#Tensor) objects.  Functions fall into several types of
categories:
  * [constructors](#torch.construction.dok) like [zeros](#torch.zeros), [ones](#torch.ones)
  * extractors like  [diag](#torch.diag)  and [triu](#torch.triu),
  * [Element-wise](#torch.elementwise.dok) mathematical operations like [abs](#torch.abs) and [pow](#torch.pow),
  * [BLAS](#torch.basicoperations.dok) operations,
  * [column or row-wise operations](#torch.columnwise.dok) like [sum](#torch.sum) and [max](#torch.max),
  * [matrix-wide operations](#torch.matrixwide.dok) like [trace](#torch.trace) and [norm](#torch.norm).
  * [Convolution and cross-correlation](#torch.conv.dok) operations like [conv2](#torch.conv2).
  * [Basic linear algebra operations](#torch.linalg.dok) like [eigen value/vector calculation](#torch.eig).
  * [Logical Operations on Tensors](#torch.logical.dok).

By default, all operations allocate a new tensor to return the
result. However, all functions also support passing the target
tensor(s) as the first argument(s), in which case the target tensor(s)
will be resized accordingly and filled with result. This property is
especially useful when one wants have tight control over when memory
is allocated.

The torch package adopts the same concept, so that calling a function
directly on the tensor itself using an object-oriented syntax is
equivalent to passing the tensor as the optional resulting tensor. The
following two calls are equivalent.

```lua
torch.log(x,x)
x:log()
```

Similarly, `torch.conv2` function can be used in the following manner.
```lua

x = torch.rand(100,100)
k = torch.rand(10,10)
res1 = torch.conv2(x,k)

res2 = torch.Tensor()
torch.conv2(res2,x,k)

=res2:dist(res1)
0

```

The advantage of second case is, same `res2` tensor can be used successively in a loop without any new allocation.

```lua
-- no new memory allocations...
for i=1,100 do
    torch.conv2(res2,x,k)
end
=res2:dist(res1)
0
```

<a name="torch.construction.dok"/>
## Construction or extraction functions ##

<a name="torch.cat"/>
### [res] torch.cat( [res,] x_1, x_2, [dimension] ) ###
<a name="torch.Tensor.cat"/>
`x=torch.cat(x_1,x_2,[dimension])` returns a tensor `x` which is the concatenation of tensors x_1 and x_2 along dimension `dimension`. 

If `dimension` is not specified it is the last dimension.

The other dimensions of x_1 and x_2 have to be equal.

Examples:
```lua
> print(torch.cat(torch.ones(3),torch.zeros(2)))

 1
 1
 1
 0
 0
[torch.Tensor of dimension 5]


> print(torch.cat(torch.ones(3,2),torch.zeros(2,2),1))

 1  1
 1  1
 1  1
 0  0
 0  0
[torch.DoubleTensor of dimension 5x2]


> print(torch.cat(torch.ones(2,2),torch.zeros(2,2),1))
 1  1
 1  1
 0  0
 0  0
[torch.DoubleTensor of dimension 4x2]

> print(torch.cat(torch.ones(2,2),torch.zeros(2,2),2))
 1  1  0  0
 1  1  0  0
[torch.DoubleTensor of dimension 2x4]


> print(torch.cat(torch.cat(torch.ones(2,2),torch.zeros(2,2),1),torch.rand(3,2),1))

 1.0000  1.0000
 1.0000  1.0000
 0.0000  0.0000
 0.0000  0.0000
 0.3227  0.0493
 0.9161  0.1086
 0.2206  0.7449
[torch.DoubleTensor of dimension 7x2]

```


<a name="torch.diag"/>
### [res] torch.diag([res,] x [,k]) ###
<a name="torch.Tensor.diag"/>

`y=torch.diag(x)` when x is of dimension 1 returns a diagonal matrix with diagonal elements constructed from x.

`y=torch.diag(x)` when x is of dimension 2 returns a tensor of dimension 1
with elements constructed from the diagonal of x.

`y=torch.diag(x,k)` returns the k-th diagonal of x,
wher k=0 is the main diagonal, k>0 is above the main diagonal and k<0 
is below the main diagonal.

<a name="torch.eye"/>
### [res] torch.eye([res,] n [,m]) ###
<a name="torch.Tensor.eye"/>

`y=torch.eye(n)` returns the n-by-n identity matrix.

`y=torch.eye(n,m)` returns an n-by-m identity matrix with ones on the diagonal and zeros elsewhere.


<a name="torch.linspace"/>
### [res] torch.linspace([res,] x1, x2, [,n]) ###
<a name="torch.Tensor.linspace"/>

`y=torch.linspace(x1,x2)` returns a one-dimensional tensor of size 100 equally spaced points between x1 and x2.

`y=torch.linspace(x1,x2,n)` returns a one-dimensional tensor of n equally spaced points between x1 and x2.


<a name="torch.logspace"/>
### [res] torch.logspace([res,] x1, x2, [,n]) ###
<a name="torch.Tensor.logspace"/>

`y=torch.logspace(x1,x2)` returns a one-dimensional tensor of 50 logarithmically eqally spaced points between x1 and x2.

`y=torch.logspace(x1,x2,n)` returns a one-dimensional tensor of n logarithmically equally spaced points between x1 and x2.

<a name="torch.multinomial"/>
### [res] torch.multinomial([res,], p, n, [,flag]) ###
<a name="torch.Tensor.multinomial"/>

`y=torch.multinomial(p,n)` returns a tensor `y` where each row contains
`n` indices sampled __with replacement__ (`flag=true`) from the 
[multinomial probability distribution](http://en.wikipedia.org/wiki/Multinomial_distribution)
located in the corresponding row of tensor `p`. 

The rows of `p` need not sum to one (in which case we use the values as weights), 
but should be non-negative and have a non-zero sum. 
Indices are ordered from left to right according to 
when each was sampled (first samples are placed in first column). 

If `p` is an `MxN` matrix, `y` is an `Mxn` matrix.

If `p` is a vector of size `M`, `y` is a vector size `n`.

`y=torch.multinomial(p,n,false)` is like the above, except samples are drawn
__without replacement__. In other words, when a sample index is drawn for a row, it 
cannot be drawn again for that row. This implies the constraint `n <= N`.

<a name="torch.ones"/>
### [res] torch.ones([res,] m [,n...]) ###
<a name="torch.Tensor.ones"/>

`y=torch.ones(n)` returns a one-dimensional tensor of size n filled with ones.

`y=torch.ones(m,n)` returns a mxn tensor filled with ones.

For more than 4 dimensions, you can use a storage as argument:
`y=torch.ones(torch.LongStorage{m,n,k,l,o})`.

<a name="torch.rand"/>
### [res] torch.rand([res,] m [,n...]) ###
<a name="torch.Tensor.rand"/>

`y=torch.rand(n)` returns a one-dimensional tensor of size n filled with random numbers from a uniform distribution on the interval (0,1).

`y=torch.rand(m,n)` returns a mxn tensor of random numbers from a uniform distribution on the interval (0,1).

For more than 4 dimensions, you can use a storage as argument:
`y=torch.rand(torch.LongStorage{m,n,k,l,o})`

<a name="torch.randn"/>
### [res] torch.randn([res,] m [,n...]) ###
<a name="torch.Tensor.randn"/>

`y=torch.randn(n)` returns a one-dimensional tensor of size n filled with random numbers from a normal distribution with mean zero and variance one.

`y=torch.randn(m,n)` returns a mxn tensor of random numbers from a normal distribution with mean zero and variance one.

For more than 4 dimensions, you can use a storage as argument:
`y=torch.rand(torch.LongStorage{m,n,k,l,o})`

<a name="torch.range"/>
### [res] torch.range([res,] x, y [,step]) ###
<a name="torch.Tensor.range"/>

`y=torch.range(x,y)` returns a tensor of size (int)(y-x)+1 with values
from x to y with step 1. You can modifiy the default step with:
`y=torch.range(x,y,step)`

```lua
> print(torch.range(2,5))

 2
 3
 4
 5
[torch.Tensor of dimension 4]
```

`y=torch.range(n,m,incr)` returns a tensor filled in range n to m with incr increments.
```lua
print(torch.range(2,5,1.2))
 2.0000
 3.2000
 4.4000
[torch.DoubleTensor of dimension 3]
```

<a name="torch.randperm"/>
### [res] torch.randperm([res,] n) ###
<a name="torch.Tensor.randperm"/>

`y=torch.randperm(n)` returns a random permutation of integers from 1 to n.

<a name="torch.reshape"/>
### [res] torch.reshape([res,] x, m [,n...]) ###
<a name="torch.Tensor.reshape"/>

`y=torch.reshape(x,m,n)` returns a new mxn tensor y whose elements
are taken rowwise from x, which must have m*n elements. The elements are copied into the new tensor.

For more than 4 dimensions, you can use a storage:
`y=torch.reshape(x,torch.LongStorage{m,n,k,l,o})`

<a name="torch.tril"/>
### [res] torch.tril([res,] x [,k]) ###
<a name="torch.Tensor.tril"/>

`y=torch.tril(x)` returns the lower triangular part of x, the other elements of y are set to 0.

`torch.tril(x,k)` returns the elements on and below the k-th diagonal of x as non-zero.   k=0 is the main diagonal, k>0 is above the main diagonal and k<0 
is below the main diagonal.

<a name="torch.triu"/>
### [res] torch.triu([res,] x, [,k]) ###
<a name="torch.Tensor.triu"/>

`y=torch.triu(x)` returns the upper triangular part of x,
the other elements of y are set to 0.

`torch.triu(x,k)` returns the elements on and above the k-th diagonal of x as non-zero.   k=0 is the main diagonal, k>0 is above the main diagonal and k<0 
is below the main diagonal.

<a name="torch.zeros"/>
### [res] torch.zeros([res,] x) ###
<a name="torch.Tensor.zeros"/>

`y=torch.zeros(n)` returns a one-dimensional tensor of size n filled with zeros.

`y=torch.zeros(m,n)` returns a mxn tensor filled with zeros.

For more than 4 dimensions, you can use a storage:
`y=torch.zeros(torch.LongStorage{m,n,k,l,o})`

<a name="torch.elementwise.dok"/>
### Element-wise Mathematical Operations ###

<a name="torch.abs"/>
### [res] torch.abs([res,] x) ###
<a name="torch.Tensor.abs"/>

`y=torch.abs(x)` returns a new tensor with the absolute values of the elements of x.

`x:abs()` replaces all elements in-place with the absolute values of the elements of x.

<a name="torch.acos"/>
### [res] torch.acos([res,] x) ###
<a name="torch.Tensor.acos"/>

`y=torch.acos(x)` returns a new tensor with the arcosine of the elements of x.

`x:acos()` replaces all elements in-place with the arcosine of the elements of x.

<a name="torch.asin"/>
### [res] torch.asin([res,] x) ###
<a name="torch.Tensor.asin"/>

`y=torch.asin(x)` returns a new tensor with the arcsine  of the elements of x.

`x:asin()` replaces all elements in-place with the arcsine  of the elements of x.

<a name="torch.atan"/>
### [res] torch.atan([res,] x) ###
<a name="torch.Tensor.atan"/>

`y=torch.atan(x)` returns a new tensor with the arctangent of the elements of x.

`x:atan()` replaces all elements in-place with the arctangent of the elements of x.

<a name="torch.ceil"/>
### [res] torch.ceil([res,] x) ###
<a name="torch.Tensor.ceil"/>

`y=torch.ceil(x)` returns a new tensor with the values of the elements of x rounded up to the nearest integers.

`x:ceil()` replaces all elements in-place with the values of the elements of x rounded up to the nearest integers.

<a name="torch.cos"/>
### [res] torch.cos([res,] x) ###
<a name="torch.Tensor.cos"/>

`y=torch.cos(x)` returns a new tensor with the cosine of the elements of x.

`x:cos()` replaces all elements in-place with the cosine of the elements of x.

<a name="torch.cosh"/>
### [res] torch.cosh([res,] x) ###
<a name="torch.Tensor.cosh"/>

`y=torch.cosh(x)` returns a new tensor with the hyberbolic cosine of the elements of x.

`x:cosh()` replaces all elements in-place with the hyberbolic cosine of the elements of x.

<a name="torch.exp"/>
### [res] torch.exp([res,] x) ###
<a name="torch.Tensor.exp"/>

`y=torch.exp(x)` returns, for each element in x,  e (the base of natural logarithms) raised to the power of the element in x.

`x:exp()` returns, for each element in x,  e (the base of natural logarithms) raised to the power of the element in x.

<a name="torch.floor"/>
### [res] torch.floor([res,] x) ###
<a name="torch.Tensor.floor"/>

`y=torch.floor(x)` returns a new tensor with the values of the elements of x rounded down to the nearest integers.

`x:floor()` replaces all elements in-place with the values of the elements of x rounded down to the nearest integers.

<a name="torch.log"/>
### [res] torch.log([res,] x) ###
<a name="torch.Tensor.log"/>

`y=torch.log(x)` returns a new tensor with the natural logarithm of the elements of x.

`x:log()` replaces all elements in-place with the natural logarithm of the elements of x.

<a name="torch.log1p"/>
### [res] torch.log1p([res,] x) ###
<a name="torch.Tensor.log1p"/>

`y=torch.log1p(x)` returns a new tensor with the natural logarithm of the elements of x+1.

`x:log1p()` replaces all elements in-place with the natural logarithm of the elements of x+1.
This function is more accurate than [log()](#torch.log) for small values of x.

<a name="torch.pow"/>
### [res] torch.pow([res,] x) ###
<a name="torch.Tensor.pow"/>

`y=torch.pow(x,n)` returns a new tensor with the elements of x to the power of n.

`x:pow(n)` replaces all elements in-place with the elements of x to the power of n.

<a name="torch.sin"/>
### [res] torch.sin([res,] x) ###
<a name="torch.Tensor.sin"/>

`y=torch.sin(x)` returns a new tensor with the sine  of the elements of x.

`x:sin()` replaces all elements in-place with the sine  of the elements of x.

<a name="torch.sinh"/>
### [res] torch.sinh([res,] x) ###
<a name="torch.Tensor.sinh"/>

`y=torch.sinh(x)` returns a new tensor with the hyperbolic sine of the elements of x.

`x:sinh()` replaces all elements in-place with the hyperbolic sine of the elements of x.

<a name="torch.sqrt"/>
### [res] torch.sqrt([res,] x) ###
<a name="torch.Tensor.sqrt"/>

`y=torch.sqrt(x)` returns a new tensor with the square root of the elements of x.

`x:sqrt()` replaces all elements in-place with the square root of the elements of x.

<a name="torch.tan"/>
### [res] torch.tan([res,] x) ###
<a name="torch.Tensor.tan"/>

`y=torch.tan(x)` returns a new tensor with the tangent of the elements of x.

`x:tan()` replaces all elements in-place with the tangent of the elements of x.

<a name="torch.tanh"/>
### [res] torch.tanh([res,] x) ###
<a name="torch.Tensor.tanh"/>

`y=torch.tanh(x)` returns a new tensor with the hyperbolic tangent of the elements of x.

`x:tanh()` replaces all elements in-place with the hyperbolic tangent of the elements of x.

<a name="torch.basicoperations.dok"/>
## Basic operations ##

In this section, we explain basic mathematical operations for Tensors.

<a name="torch.Tensor.add"/>
### [res] torch.add([res,] tensor, value) ###
<a name="torch.add"/>

Add the given value to all elements in the tensor.

`y=torch.add(x,value)` returns a new tensor.

`x:add(value)` add `value` to all elements in place.

<a name="torch.Tensor.add"/>
### [res] torch.add([res,] tensor1, tensor2) ###
<a name="torch.add"/>

Add `tensor1` to `tensor2` and put result into `res`. The number
of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:add(y)
> = x

 5  5
 5  5
[torch.Tensor of dimension 2x2]
```

`y=torch.add(a,b)` returns a new tensor.

`torch.add(y,a,b)` puts `a+b` in `y`.

`a:add(b)` accumulates all elements of `b` into `a`.

`y:add(a,b)` puts `a+b` in `y`.

<a name="torch.Tensor.add"/>
### [res] torch.add([res,] tensor1, value, tensor2) ###
<a name="torch.add"/>

Multiply elements of `tensor2` by the scalar `value` and add it to
`tensor1`. The number of elements must match, but sizes do not
matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:add(2, y)
> = x

 8  8
 8  8
[torch.Tensor of dimension 2x2]
```

`x:add(value,y)` multiply-accumulates values of `y` into `x`.

`z:add(x,value,y)` puts the result of `x + value*y` in `z`.

`torch.add(x,value,y)` returns a new tensor `x + value*y`.

`torch.add(z,x,value,y)` puts the result of `x + value*y` in `z`.

<a name="torch.Tensor.mul"/>
### [res] torch.mul([res,] tensor1, value) ###
<a name="torch.mul"/>

Multiply all elements in the tensor by the given `value`.

`z=torch.mul(x,2)` will return a new tensor with the result of `x*2`.

`torch.mul(z,x,2)` will put the result of `x*2` in `z`.

`x:mul(2)` will multiply all elements of `x` with `2` in-place.

`z:mul(x,2)` with put the result of `x*2` in `z`.

<a name="torch.Tensor.cmul"/>
### [res] torch.cmul([res,] tensor1, tensor2) ###
<a name="torch.cmul"/>

Element-wise multiplication of `tensor1` by `tensor2`. The number
of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:cmul(y)
> = x

 6  6
 6  6
[torch.Tensor of dimension 2x2]
```

`z=torch.cmul(x,y)` returns a new tensor.

`torch.cmul(z,x,y)` puts the result in `z`.

`y:cmul(x)` multiplies all elements of `y` with corresponding elements of `x`.

`z:cmul(x,y)` puts the result in `z`.

<a name="torch.Tensor.addcmul"/>
### [res] torch.addcmul([res,] x [,value], tensor1, tensor2) ###
<a name="torch.addcmul"/>

Performs the element-wise multiplication of `tensor1` by `tensor2`,
multiply the result by the scalar `value` (1 if not present) and add it
to `x`. The number of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> z = torch.Tensor(2,2):fill(5)
> x:addcmul(2, y, z)
> = x

 32  32
 32  32
[torch.Tensor of dimension 2x2]
```

`z:addcmul(value,x,y)` accumulates the result in `z`.

`torch.addcmul(z,value,x,y)` returns a new tensor with the result.

`torch.addcmul(z,z,value,x,y)` puts the result in `z`.

<a name="torch.Tensor.div"/>
### [res] torch.div([res,] tensor, value) ###
<a name="torch.div"/>

Divide all elements in the tensor by the given `value`.

`z=torch.div(x,2)` will return a new tensor with the result of `x/2`.

`torch.div(z,x,2)` will put the result of `x/2` in `z`.

`x:div(2)` will divide all elements of `x` with `2` in-place.

`z:div(x,2)` with put the result of `x/2` in `z`.

<a name="torch.Tensor.cdiv"/>
### [res] torch.cdiv([res,] tensor1, tensor2) ###
<a name="torch.cdiv"/>

Performs the element-wise division of `tensor1` by `tensor2`.  The
number of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(1)
> y = torch.Tensor(4)        
> for i=1,4 do y[i] = i end
> x:cdiv(y)
> = x

 1.0000  0.3333
 0.5000  0.2500
[torch.Tensor of dimension 2x2]
```

`z=torch.cdiv(x,y)` returns a new tensor.

`torch.cdiv(z,x,y)` puts the result in `z`.

`y:cdiv(x)` divides all elements of `y` with corresponding elements of `x`.

`z:cdiv(x,y)` puts the result in `z`.

<a name="torch.Tensor.addcdiv"/>
### [res] torch.addcdiv([res,] x [,value], tensor1, tensor2) ###
<a name="torch.addcdiv"/>

Performs the element-wise division of `tensor1` by `tensor1`, 
multiply the result by the scalar `value` and add it to `x`. 
The number of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(1)
> y = torch.Tensor(4)
> z = torch.Tensor(2,2):fill(5)
> for i=1,4 do y[i] = i end
> x:addcdiv(2, y, z)
> = x

 1.4000  2.2000
 1.8000  2.6000
[torch.Tensor of dimension 2x2]
```

`z:addcdiv(value,x,y)` accumulates the result in `z`.

`torch.addcdiv(z,value,x,y)` returns a new tensor with the result.

`torch.addcdiv(z,z,value,x,y)` puts the result in `z`.

<a name="torch.Tensor.dot"/>
### [number] torch.dot(tensor1,tensor2) ###
<a name="torch.dot"/>

Performs the dot product between `tensor` and self. The number of
elements must match: both tensors are seen as a 1D vector.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> = x:dot(y)
24
```

`torch.dot(x,y)` returns dot product of `x` and `y`.
`x:dot(y)` returns dot product of `x` and `y`.

<a name="torch.Tensor.addmv"/>
### [res] torch.addmv([res,] [v1,] vec1, [v2,] mat, vec2) ###
<a name="torch.addmv"/>

Performs a matrix-vector multiplication between `mat` (2D tensor)
and `vec` (1D tensor) and add it to vec1. In other words,

```
res = v1 * vec1 + v2 * mat*vec2
```

Sizes must respect the matrix-multiplication operation: if `mat` is
a `n x m` matrix, `vec2` must be vector of size `m` and `vec1` must
be a vector of size `n`.

```
> x = torch.Tensor(3):fill(0)
> M = torch.Tensor(3,2):fill(3)
> y = torch.Tensor(2):fill(2)
> x:addmv(M, y)
> = x

 12
 12
 12
[torch.Tensor of dimension 3]
```

`torch.addmv(x,y,z)` returns a new tensor with the result.

`torch.addmv(r,x,y,z)` puts the result in `r`.

`x:addmv(y,z)` accumulates `y*z` into `x`.

`r:addmv(x,y,z)` puts the result of `x+y*z` into `r`.

Optional values `v1` and `v2` are scalars that multiply 
`vec1` and `mat*vec2` respectively.

<a name="torch.Tensor.addr"/>
### [res] torch.addr([res,] [v1,] mat, [v2,] vec1, vec2) ###
<a name="torch.addr"/>

Performs the outer-product between `vec1` (1D tensor) and `vec2` (1D tensor).
In other words,

```
res_ij = v1 * mat_ij + v2 * vec1_i * vec2_j
```

If `vec1` is a vector of size `n` and `vec2` is a vector of size `m`, 
then mat must be a matrix of size `n x m`.

```
> x = torch.Tensor(3)        
> y = torch.Tensor(2)
> for i=1,3 do x[i] = i end
> for i=1,2 do y[i] = i end
> M = torch.Tensor(3, 2):zero()
> M:addr(x, y)
> = M

 1  2
 2  4
 3  6
[torch.Tensor of dimension 3x2]
```

`torch.addr(M,x,y)` returns the result in a new tensor.

`torch.addr(r,M,x,y)` puts the result in `r`.

`M:addr(x,y)` puts the result in `M`.

`r:addr(M,x,y)` puts the result in `r`.

Optional values `v1` and `v2` are scalars that multiply 
`M` and `vec1 [out] vec2` respectively.


<a name="torch.Tensor.addmm"/>
### [res] torch.addmm([res,] [v1,] M [v2,] mat1, mat2) ###
<a name="torch.addmm"/>

Performs a matrix-matrix multiplication between `mat1` (2D tensor)
and `mat2` (2D tensor). In other words,

```
res = v1 * M + v2 * mat1*mat2
```

If `mat1` is a `n x m` matrix, `mat2` a `m x p` matrix, 
`M` must be a `n x p` matrix.

`torch.addmm(M,mat1,mat2)` returns the result in a new tensor.

`torch.addmm(r,M,mat1,mat2)` puts the result in `r`.

`M:addmm(mat1,mat2)` puts the result in `M`.

`r:addmm(M,mat1,mat2)` puts the result in `r`.

Optional values `v1` and `v2` are scalars that multiply 
`M` and `mat1 * mat2` respectively.

<a name="torch.Tensor.mv"/>
### [res] torch.mv([res,] mat, vec) ###
<a name="torch.mv"/>

Matrix vector product of `mat` and `vec`. Sizes must respect 
the matrix-multiplication operation: if `mat` is a `n x m` matrix, 
`vec` must be vector of size `m` and res must be a vector of size `n`.

`torch.mv(x,y)` puts the result in a new tensor.

`torch.mv(M,x,y)` puts the result in `M`.

`M:mv(x,y)` puts the result in `M`.

<a name="torch.Tensor.mm"/>
### [res] torch.mm([res,] mat1, mat2) ###
<a name="torch.mm"/>

Matrix matrix product of `mat1` and `mat2`. If `mat1` is a 
`n x m` matrix, `mat2` a `m x p` matrix, res must be a 
`n x p` matrix.


`torch.mm(x,y)` puts the result in a new tensor.

`torch.mm(M,x,y)` puts the result in `M`.

`M:mm(x,y)` puts the result in `M`.

<a name="torch.Tensor.ger"/>
### [res] torch.ger([res,] vec1, vec2) ###
<a name="torch.ger"/>

Outer product of `vec1` and `vec2`. If `vec1` is a vector of 
size `n` and `vec2` is a vector of size `m`, then res must 
be a matrix of size `n x m`.


`torch.ger(x,y)` puts the result in a new tensor.

`torch.ger(M,x,y)` puts the result in `M`.

`M:ger(x,y)` puts the result in `M`.


## Overloaded operators ##

It is possible to use basic mathematic operators like `+`, `-`, `/` and `*`
with tensors.  These operators are provided as a convenience. While they
might be handy, they create and return a new tensor containing the
results. They are thus not as fast as the operations available in the
[previous section](#torch.Tensor.BasicOperations.dok).

Another important point to note is that these operators are only overloaded when the first operand is a tensor. For example, this will NOT work:
```
> x = 5 + torch.rand(3)
```

### Addition and subtraction ###

You can add a tensor to another one with the `+` operator. Subtraction is done with `-`.
The number of elements in the tensors must match, but the sizes do not matter. The size
of the returned tensor will be the size of the first tensor.
```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> = x+y

 5  5
 5  5
[torch.Tensor of dimension 2x2]

> = y-x

 1
 1
 1
 1
[torch.Tensor of dimension 4]
```

A scalar might also be added or subtracted to a tensor. The scalar might be on the right or left of the operator.
```
> x = torch.Tensor(2,2):fill(2)
> = x+3

 5  5
 5  5
[torch.Tensor of dimension 2x2]

> = 3-x

 1  1
 1  1
[torch.Tensor of dimension 2x2]
```

### Negation ###

A tensor can be negated with the `-` operator placed in front:
```
> x = torch.Tensor(2,2):fill(2)
> = -x

-2 -2
-2 -2
[torch.Tensor of dimension 2x2]
```

### Multiplication ###

Multiplication between two tensors is supported with the `*` operators. The result of the multiplication
depends on the sizes of the tensors.

 - 1D and 1D: Returns the dot product between the two tensors (scalar).
 - 2D and 1D: Returns the matrix-vector operation between the two tensors (1D tensor).
 - 2D and 2D: Returns the matrix-matrix operation between the two tensors (2D tensor).
 - 4D and 2D: Returns a tensor product (2D tensor).

Sizes must be relevant for the corresponding operation.

A tensor might also be multiplied by a scalar. The scalar might be on the right or left of the operator.

Examples:
```
> M = torch.Tensor(2,2):fill(2)
> N = torch.Tensor(2,4):fill(3)
> x = torch.Tensor(2):fill(4)
> y = torch.Tensor(2):fill(5)
> = x*y -- dot product
40
> = M*x --- matrix-vector

 16
 16
[torch.Tensor of dimension 2]

> = M*N -- matrix-matrix

 12  12  12  12
 12  12  12  12
[torch.Tensor of dimension 2x4]
```


### Division ###

Only the division of a tensor by a scalar is supported with the operator `/`.
Example:
```
> x = torch.Tensor(2,2):fill(2)
> = x/3

 0.6667  0.6667
 0.6667  0.6667
[torch.Tensor of dimension 2x2]
```


<a name="torch.columnwise.dok"/>
## Column or row-wise operations  (dimension-wise operations) ##

<a name="torch.cross"/>
### [res] torch.cross([res,] a, b [,n]) ###

`y=torch.cross(a,b)` returns the cross product of the tensors a and b.
a and b must be 3 element vectors. 

`y=cross(a,b)` returns the cross product of a and b along the first dimension of length 3.

`y=cross(a,b,n)`, where a and b returns the cross
product of vectors in dimension n of a and b. 
a and b must have the same size, 
and both `a:size(n)` and `b:size(n)` must be 3.


<a name="torch.cumprod"/>
### [res] torch.cumprod([res,] x [,dim]) ###

`y=torch.cumprod(x)` returns the cumulative product of the elements
of `x`, performing the operation over the last dimension.

`y=torch.cumprod(x,n)` returns the cumulative product of the
elements of `x`, performing the operation over dimension `n`.

<a name="torch.cumsum"/>
### [res] torch.cumsum([res,] x [,dim]) ###

`y=torch.cumsum(x)` returns the cumulative sum of the elements
of `x`, performing the operation over the first dimension.

`y=torch.cumsum(x,n)` returns the cumulative sum of the elements
of `x`, performing the operation over dimension `n`.

<a name="torch.max"/>
### torch.max([resval, resind,] x [,dim]) ###

`y=torch.max(x)` returns the single largest element of `x`.

`y,i=torch.max(x,1)` returns the largest element in each column
(across rows) of `x`, and a tensor i of their corresponding indices in
`x`.

`y,i=torch.max(x,2)` performs the max operation across rows and

`y,i=torch.max(x,n)` performs the max operation over the dimension `n`.


<a name="torch.mean"/>
### [res] torch.mean([res,] x [,dim]) ###

`y=torch.mean(x)` returns the mean of all elements of `x`.

`y=torch.mean(x,1)` returns a tensor `y` of the mean of the elements in 
each column of `x`.

`y=torch.mean(x,2)` performs the mean operation for each row and

`y=torch.mean(x,n)` performs the mean operation over the dimension `n`.

<a name="torch.min"/>
### torch.min([resval, resind,] x) ###

`y=torch.min(x)` returns the single smallest element of `x`.

`y,i=torch.min(x,1)` returns the smallest element in each column
(across rows) of `x`, and a tensor i of their corresponding indices in
`x`.

`y,i=torch.min(x,2)` performs the min operation across rows and

`y,i=torch.min(x,n)` performs the min operation over the dimension `n`.


<a name="torch.prod"/>
### [res] torch.prod([res,] x [,n]) ###

`y=torch.prod(x)` returns a tensor `y` of the product of all elements in `x`. 

`y=torch.prod(x,2)` performs the prod operation for each row and

`y=torch.prod(x,n)` performs the prod operation over the dimension `n`.

<a name="torch.sort"/>
### torch.sort([resval, resind,] x [,d] [,flag]) ###

`y,i=torch.sort(x)` returns a tensor `y` where all entries
are sorted along the last dimension, in __ascending__ order. It also returns a tensor
`i` that provides the corresponding indices from `x`.

`y,i=torch.sort(x,d)` performs the sort operation along
a specific dimension `d`.

`y,i=torch.sort(x)` is therefore equivalent to
`y,i=torch.sort(x,x:dim())`

`y,i=torch.sort(x,d,true)` performs the sort operation along
a specific dimension `d`, in __descending__ order.

<a name="torch.std"/>
### [res] torch.std([res,] x, [flag] [dim]) ###

`y=torch.std(x)` returns the standard deviation of the elements of `x`.

`y=torch.std(x,dim)` performs the std operation over the dimension dim.

`y=torch.std(x,dim,false)` performs the std operation normalizing by `n-1` (this is the default).

`y=torch.std(x,dim,true)` performs the std operation normalizing by `n` instead of `n-1`.

<a name="torch.sum"/>
### [res] torch.sum([res,] x) ###

`y=torch.sum(x)` returns the sum of the elements of `x`.

`y=torch.sum(x,2)` performs the sum operation for each row and
`y=torch.sum(x,n)` performs the sum operation over the dimension `n`.

<a name="torch.var"/>
### [res] torch.var([res,] x [,dim] [,flag]) ###

`y=torch.var(x)` returns the variance of the elements of `x`.

`y=torch.var(x,dim)` performs the var operation over the dimension dim.

`y=torch.var(x,dim,false)` performs the var operation normalizing by `n-1` (this is the default).

`y=torch.var(x,dim,true)` performs the var operation normalizing by `n` instead of `n-1`.

<a name="torch.matrixwide.dok"/>
## Matrix-wide operations  (tensor-wide operations) ##

<a name="torch.norm"/>
### torch.norm(x) ###

`y=torch.norm(x)` returns the 2-norm of the tensor `x`. 

`y=torch.norm(x,p)` returns the `p`-norm of the tensor `x`. 

`y=torch.norm(x,p,dim)` returns the `p`-norms of the tensor `x` computed over the dimension dim.

<a name="torch.renorm"/>
### torch.renorm([res], x, p, dim, maxnorm) ###
Renormalizes the sub-tensors along dimension `dim` such that they do not exceed norm `maxnorm`.

`y=torch.renorm(x,p,dim,maxnorm)` returns a version of `x` with `p`-norms lower than `maxnorm` over non-`dim` dimensions. 
The `dim` argument is not to be confused with the argument of the same name in function [norm](#torch.norm). 
In this case, the `p`-norm is measured for each `i`-th sub-tensor `x:select(dim, i)`. This function is 
equivalent to (but faster than) the following:
```lua
function renorm(matrix, value, dim, maxnorm)
  local m1 = matrix:transpose(dim, 1):contiguous()
  -- collapse non-dim dimensions:
  m2 = m1:reshape(m1:size(1), m1:nElement()/m1:size(1))
  local norms = m2:norm(value,2)
  -- clip
  local new_norms = norms:clone()
  new_norms[torch.gt(norms, maxnorm)] = maxnorm
  new_norms:cdiv(norms:add(1e-7))
  -- renormalize
  m1:cmul(new_norms:expandAs(m1))
  return m1:transpose(dim, 1)
end
```

`x:renorm(p,dim,maxnorm)` returns the equivalent of `x:copy(torch.renorm(x,p,dim,maxnorm))`.

Note: this function is particularly useful as a regularizer for constraining the norm of parameter tensors. See [Hinton et al. 2012, p. 2](http://arxiv.org/pdf/1207.0580.pdf).

<a name="torch.dist"/>
### torch.dist(x,y) ###

`y=torch.dist(x,y)` returns the 2-norm of `(x-y)`. 

`y=torch.dist(x,y,p)` returns the `p`-norm of `(x-y)`. 

<a name="torch.numel"/>
### torch.numel(x) ###

`y=torch.numel(x)` returns the count of the number of elements in the matrix `x`.

<a name="torch.trace"/>
### torch.trace(x) ###

`y=torch.trace(x)` returns the trace (sum of the diagonal elements) 
of a matrix `x`. This is  equal  to the sum of the eigenvalues of `x`.
The returned value `y` is a number, not a tensor.

<a name="torch.conv.dok"/>
## Convolution Operations ##

These function implement convolution or cross-correlation of an input
image (or set of input images) with a kernel (or set of kernels). The
convolution function in Torch can handle different types of
input/kernel dimensions and produces corresponding outputs. The
general form of operations always remain the same.

<a name="torch.conv2"/>
### [res] torch.conv2([res,] x, k, ['F' or 'V']) ###
<a name="torch.Tensor.conv2"/>

This function computes 2 dimensional convolutions between ` x ` and ` k `. These operations are similar to BLAS operations when number of dimensions of input and kernel are reduced by 2.

  * ` x `  and ` k ` are 2D : convolution of a single image with a single kernel (2D output). This operation is similar to multiplication of two scalars.
  * ` x `  and ` k ` are 3D : convolution of each input slice with corresponding kernel (3D output).
  * ` x (p x m x n) ` 3D, ` k (q x p x ki x kj)` 4D : convolution of all input slices with the corresponding slice of kernel. Output is 3D ` (q x m x n) `. This operation is similar to matrix vector product of matrix ` k ` and vector ` x `.

The last argument controls if the convolution is a full (`'F'`) or valid (`'V'`) convolution. The default is `'valid'` convolution.

```lua
x=torch.rand(100,100)
k=torch.rand(10,10)
c = torch.conv2(x,k)
=c:size()

 91
 91
[torch.LongStorage of size 2]

c = torch.conv2(x,k,'F')
=c:size()

 109
 109
[torch.LongStorage of size 2]

```

<a name="torch.xcorr2"/>
### [res] torch.xcorr2([res,] x, k, ['F' or 'V']) ###
<a name="torch.Tensor.xcorr2"/>

This function operates with same options and input/output
configurations as [torch.conv2](#torch.conv2), but performs
cross-correlation of the input with the kernel ` k `.

<a name="torch.conv3"/>
### [res] torch.conv3([res,] x, k, ['F' or 'V']) ###
<a name="torch.Tensor.conv3"/>

This function computes 3 dimensional convolutions between ` x ` and ` k `. These operations are similar to BLAS operations when number of dimensions of input and kernel are reduced by 3.

  * ` x `  and ` k ` are 3D : convolution of a single image with a single kernel (3D output). This operation is similar to multiplication of two scalars.
  * ` x `  and ` k ` are 4D : convolution of each input slice with corresponding kernel (4D output).
  * ` x (p x m x n x o) ` 4D, ` k (q x p x ki x kj x kk)` 5D : convolution of all input slices with the corresponding slice of kernel. Output is 4D ` (q x m x n x o) `. This operation is similar to matrix vector product of matrix ` k ` and vector ` x `.

The last argument controls if the convolution is a full (`'F'`) or valid (`'V'`) convolution. The default is `'valid'` convolution.

```lua
x=torch.rand(100,100,100)
k=torch.rand(10,10,10)
c = torch.conv3(x,k)
=c:size()

 91
 91
 91
[torch.LongStorage of size 3]

c = torch.conv3(x,k,'F')
=c:size()

 109
 109
 109
[torch.LongStorage of size 3]

```

<a name="torch.xcorr3"/>
### [res] torch.xcorr3([res,] x, k, ['F' or 'V']) ###
<a name="torch.Tensor.xcorr3"/>

This function operates with same options and input/output
configurations as [torch.conv3](#torch.conv3), but performs
cross-correlation of the input with the kernel ` k `.

<a name="torch.linalg.dok"/>
## Eigenvalues, SVD, Linear System Solution ##

Functions in this section are implemented with an interface to LAPACK
libraries. If LAPACK libraries are not found during compilation step,
then these functions will not be available.

<a name="torch.gesv"/>
### torch.gesv([resb, resa,] b,a) ###

Solution of ` AX=B ` and `A` has to be square and non-singular.
`A` is `m x m`, `X` is `m x k`, `B` is `m x k`.

If `resb` and `resa` are given, then they will be used for
temporary storage and returning the result.

  * `resa` will contain L and U factors for `LU` factorization of `A`.
  * `resb` will contain the solution.

```lua
a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                {-6.05, -3.30,  5.36, -4.44,  1.08},
                {-0.45,  2.58, -2.70,  0.27,  9.04},
                {8.32,  2.71,  4.35,  -7.17,  2.14},
                {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()

b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                {-1.56,  4.00, -8.67,  1.75,  2.86},
                {9.81, -4.09, -4.57, -8.61,  8.99}}):t()

 =b
 4.0200 -1.5600  9.8100
 6.1900  4.0000 -4.0900
-8.2200 -8.6700 -4.5700
-7.5700  1.7500 -8.6100
-3.0300  2.8600  8.9900
[torch.DoubleTensor of dimension 5x3]

=a
 6.8000 -6.0500 -0.4500  8.3200 -9.6700
-2.1100 -3.3000  2.5800  2.7100 -5.1400
 5.6600  5.3600 -2.7000  4.3500 -7.2600
 5.9700 -4.4400  0.2700 -7.1700  6.0800
 8.2300  1.0800  9.0400  2.1400 -6.8700
[torch.DoubleTensor of dimension 5x5]


x=torch.gesv(b,a)
 =x
-0.8007 -0.3896  0.9555
-0.6952 -0.5544  0.2207
 0.5939  0.8422  1.9006
 1.3217 -0.1038  5.3577
 0.5658  0.1057  4.0406
[torch.DoubleTensor of dimension 5x3]

=b:dist(a*x)
1.1682163181673e-14

```

<a name="torch.gels"/>
### torch.gels([resb, resa,] b,a) ###

Solution of least squares and least norm  problems for a full rank ` A ` that is ` m x n`.
  * If ` n <= m `, then solve ` ||AX-B||_F `.
  * If ` n > m ` , then solve ` min ||X||_F s.t. AX=B `.

On return, first ` n ` rows of ` X ` matrix contains the solution
and the rest contains residual information. Square root of sum squares
of elements of each column of ` X ` starting at row ` n + 1 ` is
the residual for corresponding column.

```lua

a=torch.Tensor({{ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45},
                {-7.84, -0.28,  3.24,  8.09,  2.52, -5.70},
                {-4.39, -3.24,  6.27,  5.28,  0.74, -1.19},
                {4.53,  3.83, -6.64,  2.06, -2.47,  4.70}}):t()

b=torch.Tensor({{8.58,  8.26,  8.48, -5.28,  5.72,  8.93},
                {9.35, -4.43, -0.70, -0.26, -7.36, -2.52}}):t()

=a
 1.4400 -7.8400 -4.3900  4.5300
-9.9600 -0.2800 -3.2400  3.8300
-7.5500  3.2400  6.2700 -6.6400
 8.3400  8.0900  5.2800  2.0600
 7.0800  2.5200  0.7400 -2.4700
-5.4500 -5.7000 -1.1900  4.7000
[torch.DoubleTensor of dimension 6x4]

=b
 8.5800  9.3500
 8.2600 -4.4300
 8.4800 -0.7000
-5.2800 -0.2600
 5.7200 -7.3600
 8.9300 -2.5200
[torch.DoubleTensor of dimension 6x2]

x = torch.gels(b,a)
=x 
 -0.4506   0.2497 
 -0.8492  -0.9020
  0.7066   0.6323
  0.1289   0.1351
 13.1193  -7.4922
 -4.8214  -7.1361
[torch.DoubleTensor of dimension 6x2]

=b:dist(a*x:narrow(1,1,4))
17.390200628863

=math.sqrt(x:narrow(1,5,2):pow(2):sumall())
17.390200628863

```

<a name="torch.symeig"/>
### torch.symeig([rese, resv,] a, [, 'N' or 'V'] ['U' or 'L']) ###

Eigen values and eigen vectors of a symmetric real matrix ` A ` of
size ` m x m `. This function calculates all eigenvalues (and
vectors) of ` A ` such that ` A = V' diag(e) V `. Since the input
matrix ` A ` is supposed to be symmetric, only upper triangular
portion is used by default. If the 4th argument is 'L', then lower
triangular portion is used.

Third argument defines computation of eigenvectors or eigenvalues
only. If ` N `, only eignevalues are computed. If ` V `, both
eigenvalues and eigenvectors are computed.

```lua

a=torch.Tensor({{ 1.96,  0.00,  0.00,  0.00,  0.00},
                {-6.49,  3.80,  0.00,  0.00,  0.00},
                {-0.47, -6.39,  4.17,  0.00,  0.00},
		{-7.20,  1.50, -1.51,  5.70,  0.00},
		{-0.65, -6.34,  2.67,  1.80, -7.10}}):t()

=a
 1.9600 -6.4900 -0.4700 -7.2000 -0.6500
 0.0000  3.8000 -6.3900  1.5000 -6.3400
 0.0000  0.0000  4.1700 -1.5100  2.6700
 0.0000  0.0000  0.0000  5.7000  1.8000
 0.0000  0.0000  0.0000  0.0000 -7.1000
[torch.DoubleTensor of dimension 5x5]

e = torch.symeig(a)
=e
-11.0656
 -6.2287
  0.8640
  8.8655
 16.0948
[torch.DoubleTensor of dimension 5]

e,v = torch.symeig(a,'V')
=e
-11.0656
 -6.2287
  0.8640
  8.8655
 16.0948
[torch.DoubleTensor of dimension 5]

=v
-0.2981 -0.6075  0.4026 -0.3745  0.4896
-0.5078 -0.2880 -0.4066 -0.3572 -0.6053
-0.0816 -0.3843 -0.6600  0.5008  0.3991
-0.0036 -0.4467  0.4553  0.6204 -0.4564
-0.8041  0.4480  0.1725  0.3108  0.1622
[torch.DoubleTensor of dimension 5x5]

=v*torch.diag(e)*v:t()
 1.9600 -6.4900 -0.4700 -7.2000 -0.6500
-6.4900  3.8000 -6.3900  1.5000 -6.3400
-0.4700 -6.3900  4.1700 -1.5100  2.6700
-7.2000  1.5000 -1.5100  5.7000  1.8000
-0.6500 -6.3400  2.6700  1.8000 -7.1000
[torch.DoubleTensor of dimension 5x5]

=a:dist(torch.triu(v*torch.diag(e)*v:t()))
1.0219480822443e-14

```

<a name="torch.eig"/>
### torch.eig([rese, resv,] a, [, 'N' or 'V']) ###

Eigen values and eigen vectors of a general real matrix ` A ` of
size ` m x m `. This function calculates all right eigenvalues (and
vectors) of ` A ` such that ` A = V' diag(e) V `. 

Third argument defines computation of eigenvectors or eigenvalues
only. If ` N `, only eignevalues are computed. If ` V `, both
eigenvalues and eigenvectors are computed.

The eigen values returned follow [LAPACK convention](https://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-16EB5901-5644-4DA6-A332-A052309010C4.htm) and are returned as complex (real/imaginary) pairs of numbers (Nx2 dimensional tensor).

```lua

a=torch.Tensor({{ 1.96,  0.00,  0.00,  0.00,  0.00},
                {-6.49,  3.80,  0.00,  0.00,  0.00},
                {-0.47, -6.39,  4.17,  0.00,  0.00},
    {-7.20,  1.50, -1.51,  5.70,  0.00},
    {-0.65, -6.34,  2.67,  1.80, -7.10}}):t()

=a
 1.9600 -6.4900 -0.4700 -7.2000 -0.6500
 0.0000  3.8000 -6.3900  1.5000 -6.3400
 0.0000  0.0000  4.1700 -1.5100  2.6700
 0.0000  0.0000  0.0000  5.7000  1.8000
 0.0000  0.0000  0.0000  0.0000 -7.1000
[torch.DoubleTensor of dimension 5x5]

b = a+torch.triu(a,1):t()
=b

  1.9600 -6.4900 -0.4700 -7.2000 -0.6500
 -6.4900  3.8000 -6.3900  1.5000 -6.3400
 -0.4700 -6.3900  4.1700 -1.5100  2.6700
 -7.2000  1.5000 -1.5100  5.7000  1.8000
 -0.6500 -6.3400  2.6700  1.8000 -7.1000
[torch.DoubleTensor of dimension 5x5]

e = torch.eig(b)
=e
 16.0948   0.0000
-11.0656   0.0000
 -6.2287   0.0000
  0.8640   0.0000
  8.8655   0.0000
[torch.DoubleTensor of dimension 5x2]

e,v = torch.eig(b,'V')
=e
 16.0948   0.0000
-11.0656   0.0000
 -6.2287   0.0000
  0.8640   0.0000
  8.8655   0.0000
[torch.DoubleTensor of dimension 5x2]

=v
-0.4896  0.2981 -0.6075 -0.4026 -0.3745
 0.6053  0.5078 -0.2880  0.4066 -0.3572
-0.3991  0.0816 -0.3843  0.6600  0.5008
 0.4564  0.0036 -0.4467 -0.4553  0.6204
-0.1622  0.8041  0.4480 -0.1725  0.3108
[torch.DoubleTensor of dimension 5x5]

=v*torch.diag(e:select(2,1))*v:t()
 1.9600 -6.4900 -0.4700 -7.2000 -0.6500
-6.4900  3.8000 -6.3900  1.5000 -6.3400
-0.4700 -6.3900  4.1700 -1.5100  2.6700
-7.2000  1.5000 -1.5100  5.7000  1.8000
-0.6500 -6.3400  2.6700  1.8000 -7.1000
[torch.DoubleTensor of dimension 5x5]

=b:dist(v*torch.diag(e:select(2,1))*v:t())
3.5423944346685e-14

```

<a name="torch.svd"/>
### torch.svd([resu, ress, resv] a, [, 'S' or 'A']) ###

Singular value decomposition of a real matrix `A` of size `n x m`
such that `A = USV'*T`. The call to `svd` returns `U,S,V`.

The last argument, if it is string, represents the number of singular
values to be computed. `'S'` stands for 'some' and `'A'` stands for 'all'.

```lua

a=torch.Tensor({{8.79,  6.11, -9.15,  9.57, -3.49,  9.84},
		{9.93,  6.91, -7.93,  1.64,  4.02,  0.15},
		{9.83,  5.04,  4.86,  8.83,  9.80, -8.99},
		{5.45, -0.27,  4.85,  0.74, 10.00, -6.02},
		{3.16,  7.98,  3.01,  5.80,  4.27, -5.31}}):t()
=a
  8.7900   9.9300   9.8300   5.4500   3.1600
  6.1100   6.9100   5.0400  -0.2700   7.9800
 -9.1500  -7.9300   4.8600   4.8500   3.0100
  9.5700   1.6400   8.8300   0.7400   5.8000
 -3.4900   4.0200   9.8000  10.0000   4.2700
  9.8400   0.1500  -8.9900  -6.0200  -5.3100

u,s,v = torch.svd(a)

=u
-0.5911  0.2632  0.3554  0.3143  0.2299
-0.3976  0.2438 -0.2224 -0.7535 -0.3636
-0.0335 -0.6003 -0.4508  0.2334 -0.3055
-0.4297  0.2362 -0.6859  0.3319  0.1649
-0.4697 -0.3509  0.3874  0.1587 -0.5183
 0.2934  0.5763 -0.0209  0.3791 -0.6526
[torch.DoubleTensor of dimension 6x5]

=s
 27.4687
 22.6432
  8.5584
  5.9857
  2.0149
[torch.DoubleTensor of dimension 5]

=v
-0.2514  0.8148 -0.2606  0.3967 -0.2180
-0.3968  0.3587  0.7008 -0.4507  0.1402
-0.6922 -0.2489 -0.2208  0.2513  0.5891
-0.3662 -0.3686  0.3859  0.4342 -0.6265
-0.4076 -0.0980 -0.4933 -0.6227 -0.4396
[torch.DoubleTensor of dimension 5x5]

=u*torch.diag(s)*v:t()
  8.7900   9.9300   9.8300   5.4500   3.1600
  6.1100   6.9100   5.0400  -0.2700   7.9800
 -9.1500  -7.9300   4.8600   4.8500   3.0100
  9.5700   1.6400   8.8300   0.7400   5.8000
 -3.4900   4.0200   9.8000  10.0000   4.2700
  9.8400   0.1500  -8.9900  -6.0200  -5.3100
[torch.DoubleTensor of dimension 6x5]

 =a:dist(u*torch.diag(s)*v:t())
2.8923773593204e-14

```

<a name="torch.inverse"/>
### torch.inverse([res,] x) ###

Computes the inverse of square matrix `x`.

`=torch.inverse(x)` returns the result as a new matrix.

`torch.inverse(y,x)` puts the result in `y`.

```lua
x=torch.rand(10,10)
y=torch.inverse(x)
z=x*y
print(z)
 1.0000 -0.0000  0.0000 -0.0000  0.0000  0.0000  0.0000 -0.0000  0.0000  0.0000
 0.0000  1.0000 -0.0000 -0.0000  0.0000  0.0000 -0.0000 -0.0000 -0.0000  0.0000
 0.0000 -0.0000  1.0000 -0.0000  0.0000  0.0000 -0.0000 -0.0000  0.0000  0.0000
 0.0000 -0.0000 -0.0000  1.0000 -0.0000  0.0000  0.0000 -0.0000 -0.0000  0.0000
 0.0000 -0.0000  0.0000 -0.0000  1.0000  0.0000  0.0000 -0.0000 -0.0000  0.0000
 0.0000 -0.0000  0.0000 -0.0000  0.0000  1.0000  0.0000 -0.0000 -0.0000  0.0000
 0.0000 -0.0000  0.0000 -0.0000  0.0000  0.0000  1.0000 -0.0000  0.0000  0.0000
 0.0000 -0.0000 -0.0000 -0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000
 0.0000 -0.0000 -0.0000 -0.0000  0.0000  0.0000 -0.0000 -0.0000  1.0000  0.0000
 0.0000 -0.0000  0.0000 -0.0000  0.0000  0.0000  0.0000 -0.0000  0.0000  1.0000
[torch.DoubleTensor of dimension 10x10]

print('Max nonzero = ', torch.max(torch.abs(z-torch.eye(10))))
Max nonzero =     2.3092638912203e-14

```

<a name="torch.logical.dok"/>
## Logical Operations on Tensors ##

These functions implement logical comparison operators that take a
tensor as input and another tensor or a number as the comparison
target.  They return a `ByteTensor` in which each element is 0 or 1
indicating if the comparison for the corresponding element was
`false` or `true` respectively.

<a name="torch.lt"/>
### torch.lt(a, b) ###

Implements `<` operator comparing each element in `a` with `b`
(if `b` is a number) or each element in `a` with corresponding element in `b`.

<a name="torch.lt"/>
### torch.le(a, b) ###

Implements `<=` operator comparing each element in `a` with `b`
(if `b` is a number) or each element in `a` with corresponding element in `b`.

<a name="torch.lt"/>
### torch.gt(a, b) ###

Implements `>` operator comparing each element in `a` with `b`
(if `b` is a number) or each element in `a` with corresponding element in `b`.

<a name="torch.lt"/>
### torch.ge(a, b) ###

Implements `>=` operator comparing each element in `a` with `b`
(if `b` is a number) or each element in `a` with corresponding element in `b`.

<a name="torch.lt"/>
### torch.eq(a, b) ###

Implements `==` operator comparing each element in `a` with `b`
(if `b` is a number) or each element in `a` with corresponding element in `b`.

<a name="torch.lt"/>
### torch.ne(a, b) ###

Implements `!=` operator comparing each element in `a` with `b`
(if `b` is a number) or each element in `a` with corresponding element in `b`.

### torch.all(a) ###
### torch.any(a) ###

Additionally, `any` and `all` logically sum a `ByteTensor` returning true
if any or all elements are logically true respectively. Note that logically true
here is meant in the C sense (zero is false, non-zero is true) such as the output
of the tensor element-wise logical operations.

```lua

> a = torch.rand(10)
> b = torch.rand(10)
> =a
 0.5694
 0.5264
 0.3041
 0.4159
 0.1677
 0.7964
 0.0257
 0.2093
 0.6564
 0.0740
[torch.DoubleTensor of dimension 10]

> =b
 0.2950
 0.4867
 0.9133
 0.1291
 0.1811
 0.3921
 0.7750
 0.3259
 0.2263
 0.1737
[torch.DoubleTensor of dimension 10]

> =torch.lt(a,b)
 0
 0
 1
 0
 1
 0
 1
 1
 0
 1
[torch.ByteTensor of dimension 10]

> return torch.eq(a,b)
0
0
0
0
0
0
0
0
0
0
[torch.ByteTensor of dimension 10]

> return torch.ne(a,b)
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
[torch.ByteTensor of dimension 10]

> return torch.gt(a,b)
 1
 1
 0
 1
 0
 1
 0
 0
 1
 0
[torch.ByteTensor of dimension 10]

> a[torch.gt(a,b)] = 10
> =a
 10.0000
 10.0000
  0.3041
 10.0000
  0.1677
 10.0000
  0.0257
  0.2093
 10.0000
  0.0740
[torch.DoubleTensor of dimension 10]

> a[torch.gt(a,1)] = -1
> =a
-1.0000
-1.0000
 0.3041
-1.0000
 0.1677
-1.0000
 0.0257
 0.2093
-1.0000
 0.0740
[torch.DoubleTensor of dimension 10]

> a = torch.ones(3):byte()
> =torch.all(a)
true
> a[2] = 0
> =torch.all(a)
false
> =torch.any(a)
true
> a:zero()
> =torch.any(a)
false
```

