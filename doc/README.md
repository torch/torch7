<a name="torch.reference.dok"/>
# Torch Package Reference Manual #

__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.

<a name="torch.reference.dok"/>
## Torch Packages ##

  * Tensor Library
    * [Tensor](Tensor) defines the _all powerful_ tensor object that prvides multi-dimensional numerical arrays with type templating.
    * [Mathematical operations](maths) that are defined for the tensor object types.
    * [Storage](Storage) defines a simple storage interface that controls the underlying storage for any tensor object.
  * File I/O Interface Library
    * [File](File) is an abstract interface for common file operations.
    * [Disk File](DiskFile) defines operations on files stored on disk.
    * [Memory File](MemoryFile) defines operations on stored in RAM.
    * [Pipe File](PipeFile) defines operations for using piped commands.
    * [High-Level File operations](serialization) defines higher-level serialization functions.
  * Useful Utilities
    * [Timer](Timer) provides functionality for _measuring time_.
    * [Tester](Tester) is a generic tester framework.
    * [CmdLine](CmdLine) is a command line argument parsing utility.
    * [Random](Random) defines a random number generator package with various distributions.
    * Finally useful [utility](Utility) functions are provided for easy handling of torch tensor types and class inheritance.

