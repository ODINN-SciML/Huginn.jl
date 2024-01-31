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

function plot_analysis_flow_parameters(
        params, 
        A_values, 
        n_values, 
        rgi_ids, 
        ;iceflow_model=SIA2Dmodel, 
        mass_balance_model=TImodel1,
        
    )
    # Calculate the size of the grid
    rows = length(n_values)
    cols = length(A_values)

    if rows > 5 || cols > 5
        error("more than a 5x5 grid is not supported")
    end

    if length(rgi_ids) > 1
        error("only one glacier at a time is supported")
    end

    result = [
        generate_result(
            params, A_values[j], n_values[i], rgi_ids, 
            iceflow_model, mass_balance_model
        ) for i in 1:rows, j in 1:cols
    ]
    h_diff = [result[i,j].H[end]-result[i,j].H[1] for i in 1:rows, j in 1:cols]
    
    
    Δx = hasproperty(result[1,1], :Δx) ? result[1,1].Δx : 0
    
    #Extract longitude and latitude 
    lon = hasproperty(result[1,1], :lon) ? result[1,1].lon : "none"
    lat = hasproperty(result[1,1], :lat) ? result[1,1].lat : "none"
    

    ny, nx = size(h_diff[1,1])
    h_diff = [reverse(h_diff[i,j]', dims=2) for i in 1:rows, j in 1:cols]

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
            hm_diff = Makie.heatmap!(ax_diff, h_diff[i,j],colormap=:redsblues,colorrange=(-max_abs_value, max_abs_value))
            
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
                Makie.Colorbar(fig[1:end,cols+1],hm_diff)
            end
        end
    end
    rgi_id=rgi_ids[1]
    start_year, end_year = round.(Int, params.simulation.tspan)
    fig[0, :] = Label(fig, "Ice Thickness difference ΔH for varying A and n from $start_year to $end_year")

    fig[rows+1, :] = Label(fig, "$rgi_id - latitude = $lat ° - longitude = $lon ° - scale = $scale_number km ")
    return fig
end

function generate_result(
        params, 
        A, 
        n, 
        rgi_ids, 
        iceflow_model, 
        mass_balance_model, 
        
    )
    
    # Initialize the model using the specified or default models
    model = Model(
        iceflow = iceflow_model(params), 
        mass_balance = mass_balance_model(params)
    )
    model.iceflow.A=A
    model.iceflow.n=n
    
    # Initialize glaciers and run prediction
    glaciers = initialize_glaciers(rgi_ids, params)
    prediction = Prediction(model, glaciers, params)
    run!(prediction)

    
    # Extract the first result 
    result = prediction.results[1]
    
    
    
    return result
end


