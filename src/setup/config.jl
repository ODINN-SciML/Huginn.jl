function __init__()

    # # Create structural folders if needed
    # OGGM_path = joinpath(homedir(), "Python/OGGM_data")
    # if !isdir(OGGM_path)
    #     mkpath(OGGM_path)
    # end

end

function clean()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
 end

 function enable_multiprocessing(params::Sleipnir.Parameters)
    procs = params.simulation.workers
    if procs > 0 && params.simulation.multiprocessing
        if nprocs() < procs
            @eval begin
            addprocs($procs - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            @everywhere using Reexport
            @everywhere @reexport using Huginn
            end # @eval
        elseif nprocs() != procs && procs == 1 && !params.simulation.test_mode
            @eval begin
            rmprocs(workers(), waitfor=0)
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            end # @eval
        end
    end
    return nworkers()
end

