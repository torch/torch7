require 'torch'

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

function tests.createSharedMemFile()
  local storage, shmName = createSharedMemStorage()

  -- check that file is at /dev/shm
  tester:assert(paths.filep('/dev/shm/' .. shmName),
                'Shared memory file does not exist')

  -- collect storage and make sure that file is gone
  storage = nil
  collectgarbage()
  collectgarbage()
  tester:assert(not paths.filep('/dev/shm/' .. shmName),
                'Shared memory file still exists')
end

function tests.checkContents()
  local storage, shmName = createSharedMemStorage()
  local tensor = torch.FloatTensor(storage, 1, torch.LongStorage{storage:size()})
  tensor:copy(torch.rand(storage:size()))

  local sharedFile = torch.DiskFile('/dev/shm/'..shmName, 'r'):binary()
  for i = 1, storage:size() do
    tester:assert(sharedFile:readFloat() == storage[i], 'value is not correct')
  end
  sharedFile:close()
end

function tests.testSharing()
  -- since we are going to cast numbers into double (lua default)
  -- we specifically generate double storage
  local storage, shmName = createSharedMemStorage(nil, nil, 'DoubleStorage')
  local shmFileName = '/dev/shm/' .. shmName
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
end

tester:add(tests)
tester:run()
