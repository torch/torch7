tensor = torch.rand(2,3)
f = torch.DiskFile('tensor8.bin','w')
f:binary()
f:longSize(8)
f:writeObject(tensor)
f:close()
f = torch.DiskFile('tensor8.bin','r')
f:binary()
f:longSize(8)
tensor2 = f:readObject()
f:close()
print('Tensors are same: ',tensor:norm()==tensor2:norm())

f = torch.DiskFile('tensor4.bin','w')
f:binary()
f:longSize(4)
f:writeObject(tensor)
f:close()
f = torch.DiskFile('tensor4.bin','r')
f:binary()
f:longSize(4)
tensor2 = f:readObject()
f:close()
print('Tensors are same: ',tensor:norm()==tensor2:norm())

f = torch.MemoryFile()
f:binary()
f:longSize(8)
f:writeObject(tensor)
f:seek(1)
tensor2 = f:readObject()
f:close()
print('Tensors are same: ',tensor:norm()==tensor2:norm())

f = torch.MemoryFile()
f:binary()
f:longSize(4)
f:writeObject(tensor)
f:seek(1)
tensor2 = f:readObject()
f:close()
print('Tensors are same: ',tensor:norm()==tensor2:norm())
