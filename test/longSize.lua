require 'torch'

local tester = torch.Tester()
local tests = torch.TestSuite()

local tensor = torch.rand(2,3)

function tests.diskFileLongSize8()
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
  tester:assert(tensor:norm()==tensor2:norm())
  os.remove('tensor8.bin')
end

function tests.diskFileLongSize4()
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
  tester:assert(tensor:norm()==tensor2:norm())
  os.remove('tensor4.bin')
end

function tests.memoryFileLongSize8()
  f = torch.MemoryFile()
  f:binary()
  f:longSize(8)
  f:writeObject(tensor)
  f:seek(1)
  tensor2 = f:readObject()
  f:close()
  tester:assert(tensor:norm()==tensor2:norm())
end

function tests.memoryFileLongSize4()
  f = torch.MemoryFile()
  f:binary()
  f:longSize(4)
  f:writeObject(tensor)
  f:seek(1)
  tensor2 = f:readObject()
  f:close()
  tester:assert(tensor:norm()==tensor2:norm())
end

tester:add(tests)
tester:run()
