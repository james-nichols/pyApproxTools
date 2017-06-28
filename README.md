# ApproxTools
### A Pythonic library of tools for numerical and functional analysis, and solving PDEs

This library abstracts some of the concepts sometimes found in approximation theory and solving PDEs. In particular, it can perform abstract Hilbert space operations for useful function spaces like the Sobolev space H1.

Orthogonalisation, approximation, data-assimilation, FEM solution and various other algorithms are implemented for a variety of different settings.

At present the library has two spaces on which it can perform these operations
* The space of exactly represented trigonemtric, delta, and polynomial functions in H1
* The space of piece-wise linear functions on a triangulation in H1 in two dimensions
