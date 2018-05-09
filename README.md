# pyApproxTools

### A Pythonic library of tools for numerical and functional analysis, and solving PDEs, that treats users like adults

Mathematics is messy. But the code doesn't have to be.

**Disclaimer: _This ReadMe is more of an aspirational manifesto than an actual guide. Seeing as I'm the only one really using this code still (until it's ready to really be launched, I'm using the Readme as a kind of ideas scratch space)_**

Functions and linear operators in functional analysis boil down to vectors and matrices in any numerical approximation. This is a library that essentially mimics the behaviour of NumPy, but allows for different function spaces by extending the notion of what a dot product is. Thus the library helps build, manipulate, and solve equations of such functions and linear operators using Pythonic operations, particularly the (relatively) new matrix multiplication operator ```@```. There is planned support for a variety of dot-products, enabling analysis in a variety of function spaces common in numerical analysis, for example the Sobolev spaces ![H1](http://latex.codecogs.com/gif.latex?H^1) etc...

So what approach would we want to use if we wanted to solve a simple diffusion equation like 

![pde](http://latex.codecogs.com/gif.latex?-\nabla\cdot(a(x)\nabla&space;u(u))=f(x)) 

where ![a](http://latex.codecogs.com/gif.latex?a(x)) and ![f](http://latex.codecogs.com/gif.latex?f(x)) are given and we want to solve for ![u](http://latex.codecogs.com/gif.latex?u)?

First a discretisation, via Galerkin projection, is used. This can be a triangulation or some other orthonormal basis. This requires your input. Pre-implemented examples are FEM triangulations in ![H1](http://latex.codecogs.com/gif.latex?H^1_0), for example 

```
    function_space = pat.FEM(nodes = node_locs, norm=pat.H1_0)
```

Then we can construct the operator. The operator ```D```, defined in the library, represents ![nabla](http://latex.codecogs.com/gif.latex?\nabla) on the projection, we can 
```
    D = pat.linear_operators.nabla(function_space)
```
The functions ![a](http://latex.codecogs.com/gif.latex?a(x)) and ![f](http://latex.codecogs.com/gif.latex?f(x)) must be defined in some way on this triangulation
```
    f = pat.Function(np.ones(function_space.shape), function_space)
    a = pat.Function(np.random.random(function_space.shape), function_space)
```

Then we can define ``` A = - D @ a @ D ```, and solving the PDE is now the system ``` A @ u = f ```, which we have direct access to by calling 

``` 
    A = - D @ a @ D
    u_raw = np.linalg.solve(A.matrix_rep(), f.vector_rep()) 
    u = pat.Function(u_raw, function_space)
```

The advantages of this approach are that it gives the user control of many aspects of the problem. Different PDEs and operators all have their unique issues and instabilities. There is a variety of decompositions or techniques that trained numerical analysts might want to use, but don't want to have to re-invent the wheel of all this support machinery.

Many other PDE libraries, although highly powerful and capable of things like adaptive mesh routines, whizz-bang solvers, forget-about-the-details interpolators, hide various intermediary steps that could be of use to mathematicians.

It is hoped that this library will bridge the gap between numerical analysts who want to get their hands dirty and analyse their algorithms and methods, and practitioners who just want effective PDE solutions with clean Pythonic syntax and stright-forward error reporting.

Have you got your own tricky hyperbolic operator for which you want a custom discretisation scheme that balances gawd-knows-what terms? No problem - just bake the scheme in the the ```pat.Operator``` class using templates. You are guaranteed an operator which you can then combine with others and get the expected linear schemes, no funny business or hidden optimisation.

### Spaces, Bases, and Operators

For design purposes, we need to differentiate between these concepts. An object of type ```Space``` is analogous to what we'd traditionally call a basis in mathematical terminology: it will be a collection of functions with a _pre-defined inner-product_ between them. All vectors, the inner-products between these vectors, and then operators and other general bases, will be expressed in terms of this "fundamental basis" defined in ```Space```. To guide the intuition, the ```Space``` object will most likely contain the hat functions on a triangulation that would be used in a finite element method calculation. However, we can also allow... 

---

## The old Readme


This library abstracts some of the concepts sometimes found in approximation theory and solving PDEs. In particular, it can perform abstract Hilbert space operations for useful function spaces like the Sobolev space H1.

Orthogonalisation, approximation, data-assimilation, FEM solution and various other algorithms are implemented for a variety of different settings.

At present the library has two spaces on which it can perform these operations
* The space of exactly represented trigonemtric, delta, and polynomial functions in H1
* The space of piece-wise linear functions on a triangulation in H1 in two dimensions

## TODO List

* Plot funcs with the correct triangulation
* Full linear operators as above
