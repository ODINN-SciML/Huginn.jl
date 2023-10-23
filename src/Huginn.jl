__precompile__() # this module is safe to precompile
module Huginn

# ##############################################
# ########### PACKAGES ##############
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

### ODINN.jl dependencies ###
using Reexport
@reexport using Sleipnir
@reexport using Muninn

# ##############################################
# ############ PARAMETERS ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")

# ##############################################
# ############ PYTHON LIBRARIES ##############
# ##############################################

# We either retrieve the reexported Python libraries from Sleipnir or we start from scratch
const netCDF4::PyObject = isdefined(Sleipnir, :netCDF4) ? Sleipnir.netCDF4 : PyNULL()
const cfg::PyObject = isdefined(Sleipnir, :cfg) ? Sleipnir.cfg : PyNULL()
const utils::PyObject = isdefined(Sleipnir, :utils) ? Sleipnir.utils : PyNULL()
const workflow::PyObject = isdefined(Sleipnir, :workflow) ? Sleipnir.workflow : PyNULL()
const tasks::PyObject = isdefined(Sleipnir, :tasks) ? Sleipnir.tasks : PyNULL()
const global_tasks::PyObject = isdefined(Sleipnir, :global_tasks) ? Sleipnir.global_tasks : PyNULL()
const graphics::PyObject = isdefined(Sleipnir, :graphics) ? Sleipnir.graphics : PyNULL()
const bedtopo::PyObject = isdefined(Sleipnir, :bedtopo) ? Sleipnir.bedtopo : PyNULL()
const millan22::PyObject = isdefined(Sleipnir, :millan22) ? Sleipnir.millan22 : PyNULL()
const MBsandbox::PyObject = isdefined(Sleipnir, :MBsandbox) ? Sleipnir.MBsandbox : PyNULL()
const salem::PyObject = isdefined(Sleipnir, :salem) ? Sleipnir.salem : PyNULL()

# Essential Python libraries
const xr::PyObject = isdefined(Sleipnir, :xr) ? Sleipnir.xr : PyNULL()
const rioxarray::PyObject = isdefined(Sleipnir, :rioxarray) ? Sleipnir.rioxarray : PyNULL()
const pd::PyObject = isdefined(Sleipnir, :pd) ? Sleipnir.pd : PyNULL()

# ##############################################
# ############ HUGINN LIBRARIES ##############
# ##############################################

# Include setup for Python in case this does not already exist
include("setup/config.jl")

include("parameters/SolverParameters.jl")
# All structures and functions related to Ice flow models
include("models/iceflow/IceflowModel.jl")
# Everything related to running forward simulations of ice flow
include("simulations/predictions/Prediction.jl")

end # module