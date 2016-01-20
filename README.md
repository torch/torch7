[![Join the chat at https://gitter.im/torch/torch7](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/torch/torch7?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/torch/torch7.svg)](https://travis-ci.org/torch/torch7)

## Need help? ##

* Questions, Support, Install issues: [Google groups](https://groups.google.com/forum/#!forum/torch7)
* Reporting bugs: [torch7](https://github.com/torch/torch7/issues) [nn](https://github.com/torch/nn/issues) [cutorch](https://github.com/torch/cutorch/issues) [cunn](https://github.com/torch/cutorch/issues) [optim](https://github.com/torch/optim/issues) [threads](https://github.com/torch/threads/issues)
* Hanging out with other developers and users (strictly no install issues, no large blobs of text): [Gitter Chat](https://gitter.im/torch/torch7)

<a name="torch.reference.dok"/>
# Torch Package Reference Manual #

__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.

<a name="torch.overview.dok"/>
## Torch Packages ##

  * Tensor Library
    * [Tensor](doc/tensor.md) defines the _all powerful_ tensor object that provides multi-dimensional numerical arrays with type templating.
    * [Mathematical operations](doc/maths.md) that are defined for the tensor object types.
    * [Storage](doc/storage.md) defines a simple storage interface that controls the underlying storage for any tensor object.
  * File I/O Interface Library
    * [File](doc/file.md) is an abstract interface for common file operations.
    * [Disk File](doc/diskfile.md) defines operations on files stored on disk.
    * [Memory File](doc/memoryfile.md) defines operations on stored in RAM.
    * [Pipe File](doc/pipefile.md) defines operations for using piped commands.
    * [High-Level File operations](doc/serialization.md) defines higher-level serialization functions.
  * Useful Utilities
    * [Timer](doc/timer.md) provides functionality for _measuring time_.
    * [Tester](doc/tester.md) is a generic tester framework.
    * [CmdLine](doc/cmdline.md) is a command line argument parsing utility.
    * [Random](doc/random.md) defines a random number generator package with various distributions.
    * Finally useful [utility](doc/utility.md) functions are provided for easy handling of torch tensor types and class inheritance.

<a name="torch.links.dok"/>
## Useful Links ##

  * [Community packages](https://github.com/torch/torch7/wiki/Cheatsheet)
  * [Torch Blog](http://torch.ch/blog/)
  * [Torch Slides](https://github.com/soumith/cvpr2015/blob/master/cvpr-torch.pdf)

