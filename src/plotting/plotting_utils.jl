export plot_analysis_flow_parameters

########################################################
######## Flow-parameter analysing functions  ###########
########################################################

"""
plot_analysis_flow_parameters(tspan, A_values, n_values, rgi_ids)

Generate and plot the difference in ice thickness for a specified glacier over a time span `tspan`, for varying parameters `A_values` and `n_values`.

# Arguments
- `tspan::Tuple`: A tuple specifying the start and end years for the analysis.
- `A_values::Array`: An array of A values (rate factor in Glen's flow law) to be used in the simulation.
- `n_values::Array`: An array of n values (flow law exponent in Glen's flow law) to be used in the simulation.
- `rgi_ids::Array`: An array containing the RGI (Randolph Glacier Inventory) ID of the glacier to be analyzed.

# Returns
- `Figure`: A Makie.jl figure object containing the generated plots.

# Constraints
- Supports a maximum grid size of 5x5 (lengths of `A_values` and `n_values` each should not exceed 5).
- Only supports analysis for one glacier at a time (length of `rgi_ids` should be 1).
"""

function plot_analysis_flow_parameters(simulation::SIM, A_values, n_values) where {SIM <: Simulation}

    # Calculate the size of the grid
    rows = length(n_values)
    cols = length(A_values)


    if rows > 5 || cols > 5
        error("more than a 5x5 grid is not supported")
    end

    if length(simulation.glaciers) > 1
        error("only one glacier at a time is supported")
    end

    results = [
        generate_result(simulation, A_values[j], n_values[i]) for i in 1:rows, j in 1:cols
    ]
    h_diff = [results[i,j].H[end]-results[i,j].H[1] for i in 1:rows, j in 1:cols]


    Δx = results[1,1].Δx

    # Extract metadata about the glacier
    lon = results[1,1].lon
    lat = results[1,1].lat
    x = results[1,1].x
    y = results[1,1].y
    rgi_id = results[1,1].rgi_id
    Δx = results[1,1].Δx


    nx, ny = size(h_diff[1,1])
    h_diff = [h_diff[i,j] for i in 1:rows, j in 1:cols]

    scale_width = 0.10*nx
    scale_number = round(Δx * scale_width / 1000; digits=1)
    textsize=1.2*scale_width/max(rows,cols)

    max_abs_value = max(abs(minimum(reduce(vcat, [vec(matrix) for matrix in h_diff]))), abs(maximum(reduce(vcat, [vec(matrix) for matrix in h_diff]))))

    # Initialize the figure
    fig = Makie.Figure(size = (800, 600),layout=GridLayout(rows, cols))

    # Iterate over each combination of A and n values
    for i in 1:rows
        for j in 1:cols
            ax_diff = Makie.Axis(fig[i, j],aspect=DataAspect())
            hm_diff = Makie.heatmap!(ax_diff, Sleipnir.reverseForHeatmap(h_diff[i,j], x, y),colormap=:redsblues,colorrange=(-max_abs_value, max_abs_value))

            ax_diff.xlabel = "A= $(A_values[j]), n= $(n_values[i])"
            ax_diff.xticklabelsvisible=false
            ax_diff.yticklabelsvisible=false


            if max(rows,cols) == 5
                ax_diff.xlabelsize = 12.0
            end

            Makie.poly!(ax_diff, Rect(nx -round(0.15*nx) , round(0.075*ny), scale_width, scale_width/10), color=:black)
            Makie.text!(ax_diff, "$scale_number km",
                  position = (nx - round(0.15*nx)+scale_width/16, round(0.075*ny)+scale_width/10),
                  fontsize=textsize)

            if i == rows && j == cols
                Makie.Colorbar(fig[1:end,cols+1], hm_diff)
            end
        end
    end
    start_year, end_year = round.(Int, simulation.parameters.simulation.tspan)
    fig[0, :] = Label(fig, "Ice Thickness difference ΔH for varying A and n from $start_year to $end_year")

    fig[rows+1, :] = Label(fig, "$rgi_id - latitude = $(round(lat;digits=6)) ° - longitude = $(round(lon;digits=6)) °")
    return fig
end

"""
    generate_result(placeholder_sim::SIM, A, n) where {SIM <: Simulation}

Generate the result of a simulation by initializing the model with the specified parameters and running the simulation.

# Arguments
- `simulation::SIM`: An instance of a type that is a subtype of `Simulation`.
- `A`: The parameter to set for `simulation.model.iceflow.A`.
- `n`: The parameter to set for `simulation.model.iceflow.n`.

# Returns
- `result`: The first result from the simulation's results.
"""
function generate_result(placeholder_sim::SIM, A, n) where {SIM <: Simulation}

    # Initialize the model using the specified or default models

    if !(A isa Array)
        A = fill(A)
    end

    if !(n isa Array)
        n = fill(n)
    end

    iceflow_model = SIA2Dmodel(
        placeholder_sim.parameters;
        A = ConstantLaw{typeof(A)}(Returns(A)),
        n = ConstantLaw{typeof(n)}(Returns(n)),
    )

    model = Model(;
        mass_balance = placeholder_sim.model.mass_balance,
        iceflow = iceflow_model
    )

    simulation = Prediction(model, placeholder_sim.glaciers, placeholder_sim.parameters)
    run!(simulation)

    # Extract the first result
    result = simulation.results[1]

    return result
end


