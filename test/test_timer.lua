require 'torch'
local ffi = require 'ffi'

local tester = torch.Tester()
local tests = torch.TestSuite()

function tests.timerTime()
  local timer = torch.Timer()

  local function wait(seconds)
    if ffi.os == 'Windows' then
        os.execute(string.format('ping 127.0.0.1 -n %d > nul', seconds + 1))
    else
        os.execute(string.format('sleep %d > nul', seconds))
    end
  end

  timer:reset()
  wait(1)
  local passed_time = timer:time().real
  tester:assert(passed_time < 1.1,
               ("Too long time passed: %.1f sec >= 1.1 sec"):format(passed_time))
  tester:assert(passed_time > 0.9,
               ("Too short time passed:  %.1f sec <= 0.9 sec"):format(passed_time))

  timer:stop()
  wait(1)
  passed_time = timer:time().real
  tester:assert(passed_time < 1.1,
               ("Too long time passed: %.1f sec >= 1.1 sec"):format(passed_time))
  tester:assert(passed_time > 0.9,
               ("Too short time passed:  %.1f sec <= 0.9 sec"):format(passed_time))

  timer:resume()
  wait(1)
  passed_time = timer:time().real
  tester:assert(passed_time < 2.2,
               ("Too long time passed: %.1f sec >= 2.2 sec"):format(passed_time))
  tester:assert(passed_time > 1.8,
               ("Too short time passed:  %.1f sec <= 1.8 sec"):format(passed_time))

  timer:reset()
  wait(1)
  passed_time = timer:time().real
  tester:assert(passed_time < 1.1,
               ("Too long time passed: %.1f sec >= 1.1 sec"):format(passed_time))
  tester:assert(passed_time > 0.9,
               ("Too short time passed:  %.1f sec <= 0.9 sec"):format(passed_time))
end

tester:add(tests)
tester:run()
