<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
# Table of Content

- [PipeFile](#pipefile)
    - [torch.PipeFile(command, [mode], [quiet])](#torchpipefilecommand-mode-quiet)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<a name="torch.PipeFile.dok"></a>
# PipeFile #

Parent classes: [DiskFile](diskfile.md)

A `PipeFile` is a particular `File` which is able to perform basic read/write operations
on a command pipe. It implements all methods described in [DiskFile](diskfile.md) and [File](file.md).

The file might be open in read or write mode, depending on the parameter
`mode` (which can take the value `"r"` or `"w"`) 
given to the [torch.PipeFile(fileName, mode)](#torch.PipeFile). Read-write mode is not allowed.

<a name="torch.PipeFile"></a>
### torch.PipeFile(command, [mode], [quiet]) ###

_Constructor_ which execute `command` by opening a pipe in read or write
`mode`. Valid `mode` are `"r"` (read) or `"w"` (write). Default is read
mode.

If (and only if) `quiet` is `true`, no error will be raised in case of
problem opening the file: instead `nil` will be returned.

