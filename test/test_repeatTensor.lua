--require 'torch'

local mytester 
local torchRepeatTest = {}

function torchRepeatTest.createRepeat()
  local x = torch.FloatTensor({{1,2},{3,4}})
  local y = x:repeatTensor(1,1) 
  mytester:asserteq((x - y):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat2()
  local x = torch.FloatTensor({{1,2},{3,4}})
  local y = x:repeatTensor(2,2) 
  local z = torch.FloatTensor({{1,2,1,2},{3,4,3,4},{1,2,1,2},{3,4,3,4}})
  mytester:asserteq((y - z):norm(),0,'torch.repeatTensor value')
  mytester:asserteq((x - torch.FloatTensor({{1,2},{3,4}})):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat3()
  local x = torch.FloatTensor({{1,2},{3,4}})
  local y = torch.FloatTensor()
  y:repeatTensor(x,1,1) 
  mytester:asserteq((x - y):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat4()
  local x = torch.FloatTensor({{1,2},{3,4}})
  local y = torch.FloatTensor()
  y:repeatTensor(x,2,2) 
  local z = torch.FloatTensor({{1,2,1,2},{3,4,3,4},{1,2,1,2},{3,4,3,4}})
  mytester:asserteq((y - z):norm(),0,'torch.repeatTensor value')
  mytester:asserteq((x - torch.FloatTensor({{1,2},{3,4}})):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat5()
  local x = torch.FloatTensor({1,2})
  local y = x:repeatTensor(2,2)
  local z = torch.FloatTensor({{1,2,1,2},{1,2,1,2}})
  mytester:asserteq((y - z):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat6()
  local x = torch.FloatTensor({1,2})
  local y = torch.FloatTensor()
  y:repeatTensor(x,2,2) 
  local z = torch.FloatTensor({{1,2,1,2},{1,2,1,2}})
  mytester:asserteq((y - z):norm(),0,'torch.repeatTensor value')
  mytester:asserteq((x - torch.FloatTensor({{1,2}})):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat7()
  local x = torch.FloatTensor({1,2})
  -- we put in the repeat as a long storage
  local y = x:repeatTensor(torch.LongStorage{2,2})
  local z = torch.FloatTensor({{1,2,1,2},{1,2,1,2}})
  mytester:asserteq((y - z):norm(),0,'torch.repeatTensor value')
end

function torchRepeatTest.createRepeat8()
  local x = torch.FloatTensor({1,2})
  local y = torch.FloatTensor()
  y:repeatTensor(x,torch.LongStorage{2,2}) 
  local z = torch.FloatTensor({{1,2,1,2},{1,2,1,2}})
  mytester:asserteq((y - z):norm(),0,'torch.repeatTensor value')
  mytester:asserteq((x - torch.FloatTensor({{1,2}})):norm(),0,'torch.repeatTensor value')
end

function torch.test_repeatTensor()
   mytester = torch.Tester()
   mytester:add(torchRepeatTest)
   mytester:run()
end
