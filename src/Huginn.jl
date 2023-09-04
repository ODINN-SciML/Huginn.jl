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

# ##############################################
# ############ PARAMETERS ###############
# ##############################################

const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")

# ##############################################
# ############ PYTHON LIBRARIES ##############
# ##############################################

const netCDF4::PyObject = PyNULL()
const cfg::PyObject = PyNULL()
const utils::PyObject = PyNULL()
const workflow::PyObject = PyNULL()
const tasks::PyObject = PyNULL()
const global_tasks::PyObject = PyNULL()
const graphics::PyObject = PyNULL()
const bedtopo::PyObject = PyNULL()
const millan22::PyObject = PyNULL()
const MBsandbox::PyObject = PyNULL()
const salem::PyObject = PyNULL()

# Essential Python libraries
const xr::PyObject = PyNULL()
const rioxarray::PyObject = PyNULL()
const pd::PyObject = PyNULL()

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
