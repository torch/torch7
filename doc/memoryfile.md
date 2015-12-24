<a name="torch.MemoryFile.dok"></a>
# MemoryFile #

Parent classes: [File](file.md)

A `MemoryFile` is a particular `File` which is able to perform basic
read/write operations on a buffer in `RAM`. It implements all methods
described in [File](file.md).

The data of the `File` is contained into a `NULL` terminated
[CharStorage](storage.md).

<a name="torch.MemoryFile"></a>
### torch.MemoryFile([mode]) ###

_Constructor_ which returns a new `MemoryFile` object using `mode`. Valid
`mode` are `"r"` (read), `"w"` (write) or `"rw"` (read-write). Default is `"rw"`.


<a name="torch.MemoryFile"></a>
### torch.MemoryFile(storage, mode) ###

_Constructor_ which returns a new `MemoryFile` object, using the given
[storage](storage.md) (which must be a `CharStorage`) and `mode`. Valid
`mode` are `"r"` (read), `"w"` (write) or `"rw"` (read-write). The last character
in this storage _must_ be `NULL` or an error will be generated. This allows
to read existing memory. If used for writing, note that the `storage` might
be resized by this class if needed.

<a name="torch.MemoryFile.storage"></a>
### [CharStorage] storage() ###

Returns the [storage](storage.md) which contains all the data of the
`File` (note: this is _not_ a copy, but a _reference_ on this storage). The
size of the storage is the size of the data in the `File`, plus one, the
last character being `NULL`.

<a name="torch.MemoryFile.longSize"/></a>
### longSize([size]) ###

Longs will be written and read from the file as `size` bytes long, which
can be 0, 4 or 8. 0 means system default.
