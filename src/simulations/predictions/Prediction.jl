
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
mutable struct Prediction{CACHE} <: Simulation
    model::Sleipnir.Model
    cache::Union{CACHE, Nothing}
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    results::Vector{Results}

    function Prediction(
            model::Sleipnir.Model,
            glaciers::Vector{<:Sleipnir.AbstractGlacier},
            parameters::Sleipnir.Parameters,
            results::Vector{Results}
    )
        return new{cache_type(model)}(model, nothing, glaciers, parameters, results)
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
    label(s) = printstyled(io, rpad(s, 14); color = 183)
    sep() = printstyled(io, " · "; color = :light_black)
    field(s) = printstyled(io, s; color = :light_black)
    val(s) = print(io, s)
    hint(s) = printstyled(io, s; color = :light_black)
    check(b) = b ? "\e[32m✓\e[0m " : "\e[31m✗\e[0m "

    println(io, "Prediction")

    # glaciers
    label("  glaciers")
    n = length(prediction.glaciers)
    val("$n");
    hint(" $(n == 1 ? "glacier" : "glaciers")")
    println(io)

    # model
    label("  model")
    field("iceflow");
    print(io, " = ")
    val("$(nameof(typeof(prediction.model.iceflow)))")
    sep()
    field("mass_balance");
    print(io, " = ")
    val("$(nameof(typeof(prediction.model.mass_balance)))")
    sep()
    field("learnable");
    print(io, " = ")
    if isnothing(prediction.model.trainable_components)
        hint("(nothing)")
    else
        Base.show(io, prediction.model.trainable_components)
    end
    println(io)

    # parameters
    label("  parameters")
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
    label("  cache")
    if isnothing(prediction.cache)
        hint("(nothing)")
    else
        val("$(nameof(typeof(prediction.cache)))")
    end
    println(io)

    # results
    label("  results")
    n_results = length(prediction.results)
    if n_results == 0
        print(io, check(false));
        hint(" not yet run")
    else
        print(io, check(true));
        val(" $n_results")
        hint(" $(n_results == 1 ? "result" : "results") ready")
    end
    println(io)
end
