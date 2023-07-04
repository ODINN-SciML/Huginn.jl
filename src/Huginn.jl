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
using CairoMakie, GeoMakie
import Pkg
using Distributed
using ProgressMeter
using PyCall
using SnoopPrecompile, TimerOutputs

# To be converted to `using Sleipnir`
include("../../Sleipnir.jl/src/Sleipnir.jl")

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")


# ##############################################
# ############  PYTHON LIBRARIES  ##############
# ##############################################

@precompile_setup begin

const netCDF4::PyObject = PyNULL()
const cfg::PyObject = PyNULL()
const utils::PyObject = PyNULL()
const workflow::PyObject = PyNULL()
const tasks::PyObject = PyNULL()
const global_tasks::PyObject = PyNULL()
const graphics::PyObject = PyNULL()
# const bedtopo::PyObject = PyNULL()
# const millan22::PyObject = PyNULL()
# const MBsandbox::PyObject = PyNULL()
const salem::PyObject = PyNULL()

# Essential Python libraries
# const np::PyObject = PyNULL()
const xr::PyObject = PyNULL()
# const rioxarray::PyObject = PyNULL()
# const pd::PyObject = PyNULL()

# ##############################################
# ############  HUGINN LIBRARIES  ##############
# ##############################################

@precompile_all_calls begin

# All structures and functions related to Ice flow models
include(joinpath(Huginn.root_dir, "src/models/IceflowModel.jl"))
# Everything related to running forward simulations of ice flow
include(joinpath(Huginn.root_dir, "src/simulations/Prediction.jl"))

end # @precompile_setup
end # @precompile_all_calls
end # module

