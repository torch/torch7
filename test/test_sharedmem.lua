require 'torch'
local ffi = require 'ffi'

local tester = torch.Tester()
local tests = torch.TestSuite()

local function createSharedMemStorage(name, size, storageType)
  local storageType = storageType or 'FloatStorage'
  local shmName = name or os.tmpname():gsub('/','_')
  local isShared = true
  local isSharedMem = true
  local nElements = size or torch.random(10000, 20000)
  local storage = torch[storageType](shmName, isShared, nElements, isSharedMem)
  return storage, shmName
end

local function shmFilePath(shmName)
  return (ffi.os ~= 'Windows' and '/dev/shm/' or '') .. shmName
end

local function removeShmFile(shmFileName)
  if ffi.os == 'Windows' then
    os.remove(shmFileName)
  end
end

function tests.createSharedMemFile()
  local storage, shmName = createSharedMemStorage()
  local shmFileName = shmFilePath(shmName)

  -- check that file is at /dev/shm
  tester:assert(paths.filep(shmFileName),
                'Shared memory file exists')

  -- collect storage and make sure that file is gone
  storage = nil
  collectgarbage()
  collectgarbage()
  removeShmFile(shmFileName)
  tester:assert(not paths.filep(shmFileName),
                'Shared memory file does not exists')
end

function tests.checkContents()
  local storage, shmName = createSharedMemStorage()
  local shmFileName = shmFilePath(shmName)
  local tensor = torch.FloatTensor(storage, 1, torch.LongStorage{storage:size()})
  tensor:copy(torch.rand(storage:size()))

  local sharedFile = torch.DiskFile(shmFileName, 'r'):binary()
  for i = 1, storage:size() do
    tester:assert(sharedFile:readFloat() == storage[i], 'value is not correct')
  end
  sharedFile:close()
  removeShmFile(shmFileName)
end

function tests.testSharing()
  -- since we are going to cast numbers into double (lua default)
  -- we specifically generate double storage
  local storage, shmName = createSharedMemStorage(nil, nil, 'DoubleStorage')
  local shmFileName = shmFilePath(shmName)
  local tensor = torch.DoubleTensor(storage, 1, torch.LongStorage{storage:size()})
  tensor:copy(torch.rand(storage:size()))
  local tensorCopy = tensor.new():resizeAs(tensor):copy(tensor)

  -- access the same shared memory file as regular mapping from same process
  local storage2 = torch.DoubleStorage(shmFileName, true, storage:size())
  local tensor2 = torch.DoubleTensor(storage2, 1,
                                     torch.LongStorage{storage2:size()})
  local tensor2Copy = tensor2.new():resizeAs(tensor2):copy(tensor2)

  tester:assertTensorEq(tensorCopy, tensor2Copy, 0, 'contents don\'t match')

  -- fill tensor 1 with a random value and read from 2
  local rval = torch.uniform()
  tensor:fill(rval)
  for i = 1, tensor2:size(1) do
    tester:asserteq(tensor2[i], rval, 'content is wrong')
  end

  -- fill tensor 2 with a random value and read from 1
  local rval = torch.uniform()
  tensor2:fill(rval)
  for i = 1, tensor:size(1) do
    tester:asserteq(tensor[i], rval, 'content is wrong')
  end
  removeShmFile(shmFileName)
end

tester:add(tests)
tester:run()
