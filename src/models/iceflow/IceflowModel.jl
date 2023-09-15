export Model

# Abstract type as a parent type for ice flow models
abstract type IceflowModel <: AbstractModel end

# Subtype structure for Shallow Ice Approximation models
abstract type SIAmodel <: IceflowModel end

include("iceflow_utils.jl")
include("SIA2D/SIA2D.jl")

"""
function Model(;
    iceflow::IceflowModel
    )
Initialize Huginn flow model

"""

function Model(;
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing}
    ) where {IFM <: IceflowModel, MBM <: MBmodel}

    model = Sleipnir.Model(iceflow, mass_balance, nothing)

    return model
end