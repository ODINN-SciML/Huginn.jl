__precompile__() # this module is safe to precompile
module Huginn

# ##############################################
# ########### PACKAGES ##############
# ##############################################

using JLD2
using OrdinaryDiffEqCore
using OrdinaryDiffEq
using Base: @kwdef
using Tullio
using Infiltrator
using CairoMakie
import Pkg
using Distributed
using ProgressMeter
using Printf
using Statistics, LinearAlgebra, Polynomials
using Dates

### ODINN.jl dependencies ###
using Reexport
@reexport using Muninn # imports Sleipnir as well

# ##############################################
# ############ PARAMETERS ###############
# ##############################################

const src_dir::String = dirname(@__FILE__)
const global root_dir::String = joinpath(src_dir, "..")

# ##############################################
# ############ HUGINN LIBRARIES ##############
# ##############################################

# Include setup
include(src_dir*"/setup/config.jl")

include(src_dir*"/parameters/SolverParameters.jl")
# All structures and functions related to Ice flow models
include(src_dir*"/models/iceflow/IceflowModel.jl")
# Everything related to running forward simulations of ice flow
include(src_dir*"/simulations/predictions/Prediction.jl")

# Everything related to analytical solutions
include(src_dir*"/models/solutions/halfar.jl")

# Parameterizations
include(src_dir*"/laws/Inputs.jl")
include(src_dir*"/laws/Laws.jl")

# All the utils functions
include(src_dir*"/simulations/predictions/prediction_utils.jl")
include(src_dir*"/models/iceflow/iceflow_utils.jl")
include(src_dir*"/laws/laws_utils.jl")

# Everything related to plotting
include(src_dir*"/plotting/plotting_utils.jl")

end # module
