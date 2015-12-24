<a name="torch.random.dok"></a>
# Random Numbers #

Torch provides accurate mathematical random generation, based on
[Mersenne Twister](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html)
random number generator.

<a name=":torch.seed.dok"></a>
## Seed Handling ##

The random number generator is provided with a random seed via
[seed()](#torch.seed) when torch is being initialized. It can be
reinitialized using [seed()](#torch.seed) or [manualSeed()](#torch.manualSeed).

Initial seed can be obtained using [initialSeed()](#torch.initialSeed).

Setting a particular seed allows the user to (re)-generate a particular sequence
of random numbers. Example:

```
> torch.manualSeed(123)
> = torch.uniform()
0.69646918727085
> return  torch.uniform()
0.71295532141812
> return  torch.uniform()
0.28613933874294
> torch.manualSeed(123)
> return  torch.uniform()
0.69646918727085
> return  torch.uniform()
0.71295532141812
> return  torch.uniform()
0.28613933874294
> torch.manualSeed(torch.initialSeed())
> return  torch.uniform()
0.69646918727085
> return  torch.uniform()
0.71295532141812
> return  torch.uniform()
0.28613933874294
```

To regenerate a sequence of random numbers starting from a specific point
in the sequence, one can save the state of the random number generator
using [getRNGState()](#torch.getRNGState) and then reset the random number
generator to that state using [setRNGState()](#torch.setRNGState). Example:

```
> torch.manualSeed(123)
> = torch.uniform()
0.69646918727085
> s = torch.getRNGState()
> return  torch.uniform()
0.71295532141812
> return  torch.uniform()
0.28613933874294
> torch.setRNGState(s)
> return  torch.uniform()
0.71295532141812
> return  torch.uniform()
0.28613933874294
```

<a name="torch.seed"></a>
### [number] seed() ###

Set the seed of the random number generator using `/dev/urandom`
(on Windows the time of the computer with granularity of seconds is used).
Returns the seed obtained.

<a name="torch.manualSeed"></a>
### manualSeed(number) ###

Set the seed of the random number generator to the given `number`.

<a name="torch.initialSeed"></a>
### initialSeed() ###

Returns the initial seed used to initialize the random generator.

<a name="torch.getRNGState"></a>
### [Tensor] getRNGState() ###
Returns the current state of the random number generator as a torch.ByteTensor.
This can then be used to set the state of the RNG so that the same sequence of
random numbers is produced.

<a name="torch.setRNGState"></a>
### [Tensor] setRNGState(state) ###
Set the state of the random number generator. If `state` was obtained earlier
using `getRNGState` then the random number generator should now generate the
same numbers as it did from the point where `state` was obtained. This function
returns its argument, `state`.

<a name="torch.random"></a>
### [number] random([a], [b]) ###

Returns an unsigned 32 bit integer random number from [a,b]. By default `a` is 1 and `b` is 2^32.

<a name="torch.uniform"></a>
### [number] uniform([a],[b]) ###

Returns a random real number according to uniform distribution on [a,b). By default `a` is 0 and `b` is 1.

<a name="torch.normal"></a>
### [number] normal([mean],[stdv]) ###

Returns a random real number according to a normal distribution with the given `mean` and standard deviation `stdv`.
`stdv` must be positive.

<a name="torch.exponential"></a>
### [number] exponential(lambda) ###

Returns a random real number according to the exponential distribution
''p(x) = lambda * exp(-lambda * x)''

<a name="torch.cauchy"></a>
### [number] cauchy(median, sigma) ###

Returns a random real number according to the Cauchy distribution
''p(x) = sigma/(pi*(sigma^2 + (x-median)^2))''

<a name="torch.logNormal"></a>
### [number] logNormal(mean, stdv) ###

Returns a random real number according to the log-normal distribution, with
the given `mean` and standard deviation `stdv`.
`stdv` must be positive.

<a name="torch.geometric"></a>
### [number] geometric(p) ###

Returns a random integer number according to a geometric distribution
''p(i) = (1-p) * p^(i-1)`. `p` must satisfy `0 < p < 1''.

<a name="torch.bernoulli"></a>
### [number] bernoulli([p]) ###

Returns `1` with probability `p` and `0` with probability `1-p`. `p` must satisfy `0 <= p <= 1`.
By default `p` is equal to `0.5`.
