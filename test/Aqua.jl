function test_Aqua()
    Aqua.test_ambiguities(Huginn)
    Aqua.test_undefined_exports(Huginn)
    Aqua.test_project_extras(Huginn)
    Aqua.test_stale_deps(Huginn; ignore = [:JET, :Test, :BenchmarkTools, :Revise])
    Aqua.test_deps_compat(Huginn)
    Aqua.test_piracies(Huginn; treat_as_own = [prepare_vjp_law,
        Sleipnir.ConstantLaw, Sleipnir.NullLaw])
    Aqua.test_persistent_tasks(Huginn)
    Aqua.test_undocumented_names(Huginn; broken = true)
end
