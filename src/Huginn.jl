__precompile__() # this module is safe to precompile
module Huginn

# ##############################################
# ###########       PACKAGES     ##############
# ##############################################

using JLD2
using OrdinaryDiffEq
using Base: @kwdef
using Tullio
using Infiltrator
using Plots, PlotThemes
Plots.theme(:wong2) # sets overall theme for Plots
# using CairoMakie, GeoMakie
import Pkg
using Distributed
using ProgressMeter
using PyCall
using Reexport

### ODINN.jl dependencies  ###
@reexport using Sleipnir

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")

# ##############################################
# ############  HUGINN LIBRARIES  ##############
# ##############################################

include("parameters/SolverParameters.jl")
# All structures and functions related to Ice flow models
include("models/iceflow/IceflowModel.jl")
# Everything related to running forward simulations of ice flow
include("simulations/predictions/Prediction.jl")

end # module