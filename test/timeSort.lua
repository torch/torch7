-- gnuplot.figure(2)
-- Test torch sort, show it suffers from the problems of quicksort
-- i.e. complexity O(N^2) in worst-case of sorted list
require 'gnuplot'
local ffi = require 'ffi'

local cmd = torch.CmdLine()
cmd:option('-N', 10^7, 'Maximum array size')
cmd:option('-p',  50, 'Number of points in logspace')
cmd:option('-r', 20, 'Number of repetitions')

local options = cmd:parse(arg or {})
function main()
    local log10 = math.log10 or function(x) return math.log(x, 10) end
    local pow10 = torch.linspace(1,log10(options.N), options.p)
    local num_sizes = options.p
    local num_reps = options.r

    local old_rnd = torch.zeros(num_sizes, num_reps)
    local old_srt = torch.zeros(num_sizes, num_reps)
    local old_cst = torch.zeros(num_sizes, num_reps)
    local new_rnd = torch.zeros(num_sizes, num_reps)
    local new_srt = torch.zeros(num_sizes, num_reps)
    local new_cst = torch.zeros(num_sizes, num_reps)
    local ratio_rnd = torch.zeros(num_sizes, num_reps)
    local ratio_srt = torch.zeros(num_sizes, num_reps)
    local ratio_cst = torch.zeros(num_sizes, num_reps)

    -- Ascending sort uses new sort
    local function time_sort(x)
        collectgarbage()
        local start = os.clock()
        torch.sort(x,false)
        return (os.clock()-start)
    end

    -- Descending sort uses old sort
    local function time_old_sort(x)
        collectgarbage()
        local start = os.clock()
        torch.sort(x,true)
        return (os.clock()-start)
    end

    local benches = {
        function(i,j,n)
            -- on random
            local input = torch.rand(n)
            new_rnd[i][j] = time_sort(input:clone())
            old_rnd[i][j] = time_old_sort(input:clone())
        end,

        function(i,j,n)
            -- on sorted
            new_srt[i][j] = time_sort(torch.linspace(0,1,n))
            old_srt[i][j] = time_old_sort(torch.linspace(0,1,n):add(-1):mul(-1)) -- old_time is called on descending sort, hence the reversed input
        end,

        function(i,j,n)
            -- on constant
            new_cst[i][j] = time_sort(torch.zeros(n))
            old_cst[i][j] = time_old_sort(torch.zeros(n))
        end
    }

    local num_benches = #benches
    local num_exps = num_sizes * num_benches * num_reps

    -- Full randomization
    local perm = torch.randperm(num_exps):long()
    local perm_benches = torch.Tensor(num_exps)
    local perm_reps = torch.Tensor(num_exps)
    local perm_sizes = torch.Tensor(num_exps)

    local l = 1
    for i=1, num_sizes do
        for j=1, num_reps do
            for k=1, num_benches do
                perm_benches[ perm[l] ] = k
                perm_reps[ perm[l] ] = j
                perm_sizes[ perm[l] ] = i
                l = l+1
            end
        end
    end

    local pc = 0
    for j = 1, num_exps do
        local n = 10^pow10[perm_sizes[j]]
    --    print(string.format('rep %d / %d, bench %d, size %d, rep %d\n', j, num_exps, perm_benches[j], n, perm_reps[j]))
        if math.floor(100*j/num_exps) > pc then
            pc = math.floor(100*j/num_exps)
            io.write('.')
            if pc % 10 == 0 then
                io.write(' ' .. pc .. '%\n')
             end
            io.flush()
        end
        benches[perm_benches[j]](perm_sizes[j], perm_reps[j], n)
    end

    ratio_rnd = torch.cdiv(old_rnd:mean(2), new_rnd:mean(2))
    ratio_srt = torch.cdiv(old_srt:mean(2), new_srt:mean(2))
    ratio_cst = torch.cdiv(old_cst:mean(2), new_cst:mean(2))

    local N = pow10:clone():apply(function(x) return 10^x end)

    if ffi.os == 'Windows' then
      gnuplot.setterm('windows')
    else
      gnuplot.setterm('x11')
    end
    gnuplot.figure(1)
    gnuplot.raw('set log x; set mxtics 10')
    gnuplot.raw('set grid mxtics mytics xtics ytics')
    gnuplot.raw('set xrange [' .. N:min() .. ':' .. N:max() .. ']' )
    gnuplot.plot({'Random - new', N, new_rnd:mean(2)},
                 {'Sorted - new', N, new_srt:mean(2)},
                 {'Constant - new', N, new_cst:mean(2)},
                 {'Random - old', N, old_rnd:mean(2)},
                 {'Sorted - old', N, old_srt:mean(2)},
                 {'Constant - old', N, old_cst:mean(2)})
    gnuplot.xlabel('N')
    gnuplot.ylabel('Time (s)')
    gnuplot.figprint('benchmarkTime.png')

    gnuplot.figure(2)
    gnuplot.raw('set log x; set mxtics 10')
    gnuplot.raw('set grid mxtics mytics xtics ytics')
    gnuplot.raw('set xrange [' .. N:min() .. ':' .. N:max() .. ']' )
    gnuplot.plot({'Random', N, ratio_rnd:mean(2)},
                 {'Sorted', N, ratio_srt:mean(2)},
                 {'Constant', N, ratio_cst:mean(2)})
    gnuplot.xlabel('N')
    gnuplot.ylabel('Speed-up Factor (s)')
    gnuplot.figprint('benchmarkRatio.png')

    torch.save('benchmark.t7', {
               new_rnd=new_rnd,
               new_srt=new_srt,
               new_cst=new_cst,
               old_rnd=old_rnd,
               old_srt=old_srt,
               old_cst=old_cst,
               ratio_rnd=ratio_rnd,
               ratio_srt=ratio_srt,
               ratio_cst=ratio_cst,
               pow10 = pow10,
               num_reps = num_reps
           })
end

main()
