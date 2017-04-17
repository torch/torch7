<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
# Table of Content

- [Timer](#timer)
  - [Timer Class Constructor and Methods](#timer-class-constructor-and-methods)
    - [torch.Timer()](#torchtimer)
    - [[self] reset()](#self-reset)
    - [[self] resume()](#self-resume)
    - [[self] stop()](#self-stop)
    - [[table] time()](#table-time)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<a name="torch.Timer.dok"></a>
# Timer #

This class is able to measure time (in seconds) elapsed in a particular period. Example:
```lua
  timer = torch.Timer() -- the Timer starts to count now
  x = 0
  for i=1,1000000 do
    x = x + math.sin(x)
  end
  print('Time elapsed for 1,000,000 sin: ' .. timer:time().real .. ' seconds')
```

<a name="torch.Timer"></a>
## Timer Class Constructor and Methods ##

<a name="torch.Timer"></a>
### torch.Timer() ###

Returns a new `Timer`. The timer starts to count the time now.

<a name="torch.Timer.reset"></a>
### [self] reset() ###

Reset the timer accumulated time to `0`. If the timer was running, the timer
restarts to count the time now. If the timer was stopped, it stays stopped.

<a name="torch.Timer.resume"></a>
### [self] resume() ###

Resume a stopped timer. The timer restarts to count the time, and addition
the accumulated time with the time already counted before being stopped.

<a name="torch.Timer.stop"></a>
### [self] stop() ###

Stop the timer. The accumulated time counted until now is stored.

<a name="torch.Timer.time"></a>
### [table] time() ###

Returns a table reporting the accumulated time elapsed until now. Following the UNIX shell `time` command,
there are three fields in the table:
  * `real`: the wall-clock elapsed time.
  * `user`: the elapsed CPU time. Note that the CPU time of a threaded program sums time spent in all threads.
  * `sys`: the time spent in system usage.

