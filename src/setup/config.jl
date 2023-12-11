function __init__()

    # Create structural folders if needed
    OGGM_path = joinpath(homedir(), "Python/OGGM_data")
    if !isdir(OGGM_path)
        mkpath(OGGM_path)
    end

    try
        # Only load Python packages if not previously loaded by Sleipnir
        if cfg == PyNULL() && workflow == PyNULL() && utils == PyNULL() && MBsandbox == PyNULL() 
            println("Initializing Python libraries in Huginn...")
            copy!(netCDF4, pyimport("netCDF4"))
            copy!(cfg, pyimport("oggm.cfg"))
            copy!(utils, pyimport("oggm.utils"))
            copy!(workflow, pyimport("oggm.workflow"))
            copy!(tasks, pyimport("oggm.tasks"))
            copy!(global_tasks, pyimport("oggm.global_tasks"))
            copy!(graphics, pyimport("oggm.graphics"))
            copy!(bedtopo, pyimport("oggm.shop.bedtopo"))
            copy!(millan22, pyimport("oggm.shop.millan22"))
            copy!(MBsandbox, pyimport("MBsandbox.mbmod_daily_oneflowline"))
            copy!(salem, pyimport("salem"))
            copy!(pd, pyimport("pandas"))
            copy!(xr, pyimport("xarray"))
            copy!(rioxarray, pyimport("rioxarray"))
        end
    catch e
        @warn "It looks like you have not installed and/or activated the virtual Python environment. \n 
        Please follow the guidelines in: https://github.com/ODINN-SciML/ODINN.jl#readme"
        @warn exception=(e, catch_backtrace())
    end

end

function clean()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
 end

 function enable_multiprocessing(procs::Int)
    if procs > 0
        if nprocs() < procs
            @eval begin
            addprocs($procs - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            @everywhere using Reexport
            @everywhere @reexport using Huginn
            end # @eval
        elseif nprocs() != procs && procs == 1
            @eval begin
            rmprocs(workers(), waitfor=0)
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            end # @eval
        end
    end
    return nworkers()
end

