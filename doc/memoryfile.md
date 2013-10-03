<a name="torch.MemoryFile.dok"/>
# MemoryFile #

Parent classes: [File](File)

A `MemoryFile` is a particular `File` which is able to perform basic
read/write operations on a buffer in `RAM`. It implements all methods
described in [File](File).

The data of the this `File` is contained into a `NULL` terminated
[CharStorage](Storage).

<a name="torch.MemoryFile"/>
### torch.MemoryFile([mode]) ###

_Constructor_ which returns a new `MemoryFile` object using `mode`. Valid
`mode` are `"r"` (read), `"w"` (write) or `"rw"` (read-write). Default is `"rw"`.


<a name="torch.MemoryFile"/>
### torch.MemoryFile(storage, mode) ###

_Constructor_ which returns a new `MemoryFile` object, using the given
[storage](Storage) (which must be a `CharStorage`) and `mode`. Valid
`mode` are `"r"` (read), `"w"` (write) or `"rw"` (read-write). The last character
in this storage _must_ be `NULL` or an error will be generated. This allow
to read existing memory. If used for writing, not that the `storage` might
be resized by this class if needed. 

<a name="torch.MemoryFile.storage"/>
### [CharStorage] storage() ###

Returns the [storage](Storage) which contains all the data of the
`File` (note: this is _not_ a copy, but a _reference_ on this storage). The
size of the storage is the size of the data in the `File`, plus one, the
last character being `NULL`.

