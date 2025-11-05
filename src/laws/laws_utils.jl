import Sleipnir: prepare_vjp_law

prepare_vjp_law(simulation, law::ConstantLaw, law_cache, θ, glacier_idx) = nothing
prepare_vjp_law(simulation, law::NullLaw, law_cache, θ, glacier_idx) = nothing
prepare_vjp_law(simulation::Prediction, law::Law, law_cache, θ, glacier_idx) = nothing
