
export Prediction

# Subtype composite type for a prediction simulation
mutable struct Prediction  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    results::Vector{Results}
end

"""
    function Prediction(
        model::Sleipnir.Model,
        glaciers::Vector{Sleipnir.AbstractGlacier},
        parameters::Sleipnir.Parameters
        )
Construnctor for Prediction struct with glacier model infomation, glaciers and parameters.
Keyword arguments
=================
"""
function Prediction(
    model::Sleipnir.Model,
    glaciers::Vector{G},
    parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}

    # Build the results struct based on input values
    prediction = Prediction(model,
                            glaciers,
                            parameters,
                            Vector{Results}([]))

    return prediction
end

###############################################
################### UTILS #####################
###############################################

include("prediction_utils.jl")
