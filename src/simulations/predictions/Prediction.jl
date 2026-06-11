
export Prediction

"""
    Prediction{CACHE} <: Simulation

A mutable struct that represents a prediction simulation.

# Fields

  - `model::Sleipnir.Model`: The model used for the prediction.
  - `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the prediction.
  - `parameters::Sleipnir.Parameters`: The parameters used for the prediction.
  - `results::Vector{Results}`: A vector of results obtained from the prediction.
"""
mutable struct Prediction{
    MODEL <: Sleipnir.Model,
    CACHE,
    GLACIER <: Sleipnir.AbstractGlacier,
    PARAMS <: Sleipnir.Parameters,
    RES <: Vector{Results}
} <: Simulation
    model::MODEL
    cache::Union{CACHE, Nothing}
    glaciers::Vector{GLACIER}
    parameters::PARAMS
    results::RES

    function Prediction(
            model::Sleipnir.Model,
            glaciers::Vector{<:Sleipnir.AbstractGlacier},
            parameters::Sleipnir.Parameters,
            results::Vector{Results}
    )
        return new{typeof(model), cache_type(model), eltype(glaciers),
            typeof(parameters), typeof(results)}(
            model, nothing, glaciers, parameters, results)
    end
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
function Prediction(model::Sleipnir.Model, glaciers::Vector{G},
        parameters::Sleipnir.Parameters) where {G <: Sleipnir.AbstractGlacier}
    Prediction(model, glaciers, parameters, Results[])
end

# Display setup
Base.show(io::IO, ::MIME"text/plain", prediction::Prediction) = Base.show(io, prediction)
function Base.show(io::IO, prediction::Prediction)
    pad = 14

    println(io, "Prediction")

    # glaciers
    label(io, "  glaciers", pad)
    n = length(prediction.glaciers)
    val(io, "$n");
    hint(io, " $(n == 1 ? "glacier" : "glaciers")")
    println(io)

    # model
    label(io, "  model", pad)
    field(io, "iceflow");
    print(io, " = ")
    val(io, "$(nameof(typeof(prediction.model.iceflow)))")
    sep(io)
    field(io, "mass_balance");
    print(io, " = ")
    val(io, "$(nameof(typeof(prediction.model.mass_balance)))")
    sep(io)
    field(io, "learnable");
    print(io, " = ")
    if isnothing(prediction.model.trainable_components)
        hint(io, "(nothing)")
    else
        Base.show(io, prediction.model.trainable_components)
    end
    println(io)

    # parameters
    label(io, "  parameters", pad)
    println(io)
    # Capture the Parameters show output and re-indent each line
    params_str = sprint(show, prediction.parameters)
    for line in split(params_str, "\n")
        isempty(line) && continue
        # Skip the "Parameters" header line since the label already identifies it
        occursin(r"^Parameters$", line) && continue
        printstyled(io, "    "; color = :light_black)
        println(io, line)
    end

    # cache
    label(io, "  cache", pad)
    if isnothing(prediction.cache)
        hint(io, "(nothing)")
    else
        val(io, "$(nameof(typeof(prediction.cache)))")
    end
    println(io)

    # results
    label(io, "  results", pad)
    n_results = length(prediction.results)
    if n_results == 0
        print(io, check(false));
        hint(io, " not yet run")
    else
        print(io, check(true));
        val(io, " $n_results")
        hint(io, " $(n_results == 1 ? "result" : "results") ready")
    end
    println(io)
end
