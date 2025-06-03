import Sleipnir: init_cache
export Model, IceflowModel

# Abstract type as a parent type for ice flow models
"""
    IceflowModel

An abstract type representing the base model for ice flow simulations. All specific ice flow models should subtype this abstract type.
"""
abstract type IceflowModel <: AbstractModel end

"""
    SIAmodel

An abstract type representing the Shallow Ice Approximation (SIA) models. This type is a subtype of `IceflowModel` and serves as a base for all SIA-specific models.
"""
abstract type SIAmodel <: IceflowModel end

include("iceflow_utils.jl")
include("SIA2D/SIA2D.jl")

"""
function Model(;
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing}
    ) where {IFM <: IceflowModel, MBM <: MBmodel}
    
Initialize Model at Huginn level (no machine learning model).

"""
function Model(;
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing}
    ) where {IFM <: IceflowModel, MBM <: MBmodel}

    model = Sleipnir.Model{typeof(iceflow), typeof(mass_balance), Nothing}(iceflow, mass_balance, nothing)

    return model
end
