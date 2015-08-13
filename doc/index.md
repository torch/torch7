<a name="torch.reference.dok"></a>
# Torch Package Reference Manual #

[![Join the chat at https://gitter.im/torch/torch7](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/torch/torch7?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/torch/torch7.svg)](https://travis-ci.org/torch/torch7)

__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.

<a name="torch.reference.dok"></a>
## Torch Packages ##

  * Tensor Library
    * [Tensor](tensor.md) defines the _all powerful_ tensor object that provides multi-dimensional numerical arrays with type templating.
    * [Mathematical operations](maths.md) that are defined for the tensor object types.
    * [Storage](storage.md) defines a simple storage interface that controls the underlying storage for any tensor object.
  * File I/O Interface Library
    * [File](file.md) is an abstract interface for common file operations.
    * [Disk File](diskfile.md) defines operations on files stored on disk.
    * [Memory File](memoryfile.md) defines operations on stored in RAM.
    * [Pipe File](pipefile.md) defines operations for using piped commands.
    * [High-Level File operations](serialization.md) defines higher-level serialization functions.
  * Useful Utilities
    * [Timer](timer.md) provides functionality for _measuring time_.
    * [Tester](tester.md) is a generic tester framework.
    * [CmdLine](cmdline.md) is a command line argument parsing utility.
    * [Random](random.md) defines a random number generator package with various distributions.
    * Finally useful [utility](utility.md) functions are provided for easy handling of torch tensor types and class inheritance.

