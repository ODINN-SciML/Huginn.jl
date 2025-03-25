
export Prediction

"""
    Prediction <: Simulation

A mutable struct that represents a prediction simulation.

# Fields
- `model::Sleipnir.Model`: The model used for the prediction.
- `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the prediction.
- `parameters::Sleipnir.Parameters`: The parameters used for the prediction.
- `results::Vector{Results}`: A vector of results obtained from the prediction.
"""
mutable struct Prediction  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    results::Vector{Results}
end


"""
    Prediction(model::Sleipnir.Model, glaciers::Vector{G}, parameters::Sleipnir.Parameters) where {G <: Sleipnir.AbstractGlacier}

Create a `Prediction` object using the given model, glaciers, and parameters.

# Arguments
- `model::Sleipnir.Model`: The model used for prediction.
- `glaciers::Vector{G}`: A vector of glacier objects, where each glacier is a subtype of `Sleipnir.AbstractGlacier`.
- `parameters::Sleipnir.Parameters`: The parameters used for the prediction.

# Returns
- `Prediction`: A `Prediction` object based on the input values.
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
