require 'torch'
local ffi = require 'ffi'

local tester = torch.Tester()
local tests = torch.TestSuite()

function tests.timerTime()
  local timer = torch.Timer()

  local function wait(seconds)
    if ffi.os == 'Windows' then
        os.execute(string.format('timeout %d > nul', seconds))
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
  tester:assert(passed_time < 2.1,
               ("Too long time passed: %.1f sec >= 2.1 sec"):format(passed_time))
  tester:assert(passed_time > 1.9,
               ("Too short time passed:  %.1f sec <= 1.9 sec"):format(passed_time))

  timer:reset()
  wait(1)
  local passed_time = timer:time().real
  tester:assert(passed_time < 1.1,
               ("Too long time passed: %.1f sec >= 1.1 sec"):format(passed_time))
  tester:assert(passed_time > 0.9,
               ("Too short time passed:  %.1f sec <= 0.9 sec"):format(passed_time))
end

tester:add(tests)
tester:run()
