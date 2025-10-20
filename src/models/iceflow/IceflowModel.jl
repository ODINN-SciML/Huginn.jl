export IceflowModel

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

Model = Sleipnir.Model

include("SIA2D/SIA2D.jl")
