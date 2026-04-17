
function access(x::Results, name::String)
    s = Symbol(name)
    return getproperty(x, s)
end

function plot_test_error(pred::Results, ref::Results, variable, rgi_id, atol,
        MB; path = joinpath(Huginn.root_dir, "test/plots"))
    @assert (variable == "H") || (variable == "Vx") || (variable == "Vy") "Wrong variable for plots. Needs to be either `H`, `Vx` or `Vy`."
    if !all(isapprox.(access(pred, variable)[end], access(ref, variable)[end], atol = atol))
        # @warn "Error found in PDE solve! Check plots in /test/plots⁄"
        if variable == "H"
            colour=:ice
        elseif variable == "Vx" || variable == "Vy"
            colour=:speed
        end
        MB ? tail = "MB" : tail = ""
        fig_pde, ax_pde,
        hm_pde = CairoMakie.heatmap(
            access(pred, variable)[end] .- access(ref, variable)[end],
            colormap = colour)
        ax_pde.title = "$(variable): PDE simulation - Reference simulation"
        CairoMakie.save(joinpath(path, "$(variable)_PDE_$rgi_id$tail.pdf"), fig_pde)
    end
end

function plot_test_error(pred::Tuple, ref::Dict{String, Any}, variable, rgi_id,
        atol, MB; path = joinpath(Huginn.root_dir, "test/plots"))
    @assert (variable == "H") || (variable == "Vx") || (variable == "Vy") "Wrong variable for plots. Needs to be either `H`, `Vx` or `Vy`."
    if variable == "H"
        idx=1
        colour=:ice
    elseif variable == "Vx"
        idx=2
        colour=:speed
    elseif variable == "Vy"
        idx=3
        colour=:speed
    end
    if !all(isapprox.(pred[idx], ref[variable], atol = atol))
        # @warn "Error found in PDE solve! Check plots in /test/plots⁄"
        MB ? tail = "MB" : tail = ""
        fig_ude, ax_ude,
        hm_ude = CairoMakie.heatmap(
            pred[idx] .- ref[variable],
            colormap = colour)
        ax_ude.title = "$(variable): UDE simulation - Reference simulation"
        CairoMakie.save(joinpath(path, "$(variable)_UDE_$rgi_id$tail.pdf"), fig_ude)
    end
end
