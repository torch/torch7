-- gnuplot.figure(1)
-- Test torch sort, show it suffers from the problems of quicksort
-- i.e. complexity O(N^2) in worst-case of sorted list
require 'gnuplot'

function main()
    local pow10 = torch.linspace(1,5,10)
    local old_bench_rnd = torch.zeros(pow10:numel())
    local old_bench_srt = torch.zeros(pow10:numel())
    local old_bench_cst = torch.zeros(pow10:numel())
    local new_bench_rnd = torch.zeros(pow10:numel())
    local new_bench_srt = torch.zeros(pow10:numel())
    local new_bench_cst = torch.zeros(pow10:numel())
    local ratio_rnd = torch.zeros(pow10:numel())
    local ratio_srt = torch.zeros(pow10:numel())
    local ratio_cst = torch.zeros(pow10:numel())
    local nrep = 3

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

    for j = 1,nrep do
        for i = 1,pow10:numel() do

            local new_time, old_time
            local n = 10^pow10[i]

            -- on random
            new_time = time_sort(torch.rand(n))
            old_time = time_old_sort(torch.rand(n))
            new_bench_rnd[i] = new_bench_rnd[i] + new_time/nrep
            old_bench_rnd[i] = old_bench_rnd[i] + old_time/nrep
            ratio_rnd[i] = ratio_rnd[i] + (old_bench_rnd[i]/new_bench_rnd[i])/nrep

            -- on sorted
            new_time = time_sort(torch.linspace(0,1,n))
            old_time = time_old_sort(torch.linspace(0,1,n):add(-1):mul(-1)) -- old_time is called on descending sort, hence the reversed input
            new_bench_srt[i] = new_bench_srt[i] + new_time/nrep
            old_bench_srt[i] = old_bench_srt[i] + old_time/nrep
            ratio_srt[i] = ratio_srt[i] + (old_bench_srt[i]/new_bench_srt[i])/nrep

            -- on constant
            new_time = time_sort(torch.zeros(n))
            old_time = time_old_sort(torch.zeros(n))
            new_bench_cst[i] = new_bench_cst[i] + new_time/nrep
            old_bench_cst[i] = old_bench_cst[i] + old_time/nrep
            ratio_cst[i] = ratio_cst[i] + (old_bench_cst[i]/new_bench_cst[i])/nrep
        end
        io.flush()
    end
    gnuplot.figure(1)
    gnuplot.plot({'Random - new', pow10, new_bench_rnd},
                 {'Sorted - new', pow10, new_bench_srt},
                 {'Constant - new', pow10, new_bench_cst},
                 {'Random - old', pow10, old_bench_rnd},
                 {'Sorted - old', pow10, old_bench_srt},
                 {'Constant - old', pow10, old_bench_cst})
    gnuplot.xlabel('Log10(N)')
    gnuplot.ylabel('Time (s)')
    gnuplot.figprint('benchmarkTime.png')

    gnuplot.figure(2)
    gnuplot.plot({'Random', pow10, ratio_rnd},
                 {'Sorted', pow10, ratio_srt},
                 {'Constant', pow10, ratio_cst})
    gnuplot.xlabel('Log10(N)')
    gnuplot.ylabel('Speed-up Factor (s)')
    gnuplot.figprint('benchmarkRatio.png')
end

main()
