using StructuralIdentifiability

@testset "Identifiability APIs" begin
    function shared_logistic!(du, u, p, t)
        r, K = p
        for index in eachindex(u)
            du[index] = r * u[index] * (1 - u[index] / K)
        end
    end

    times = collect(0.0:1.0:6.0)
    truth = [0.35, 25.0]
    u0 = [1.0, 2.0]
    solution = solve(ODEProblem(shared_logistic!, u0, (first(times), last(times)), truth), Tsit5(); saveat = times)
    datasets = [
        (x = times, y = [state[1] for state in solution.u], state_index = 1, residual_scale = 0.25),
        (x = times, y = [state[2] for state in solution.u], state_index = 2, residual_scale = 0.25),
    ]
    bounds = [(0.05, 1.0), (5.0, 60.0)]
    map = ObservationMap(
        "two_state_logistic",
        [:sensitive, :resistant],
        [:sensitive_count, :resistant_count];
        description = "Each assay directly measures one population.",
    )

    mapping = validate_observation_map(map, datasets)
    @test nrow(mapping) == 2
    @test all(mapping.measurement .== "state")

    starts = generate_multistarts(bounds; n_starts = 3, rng = MersenneTwister(12))
    @test length(starts) == 3
    @test all(length(start) == 2 for start in starts)
    @test all(bounds[index][1] <= start[index] <= bounds[index][2] for start in starts for index in eachindex(bounds))

    predictions = prediction_vector(shared_logistic!, datasets, u0, truth; solver = Tsit5())
    @test length(predictions) == 2 * length(times)
    @test isapprox(predictions[1], datasets[1].y[1])

    jacobian = prediction_jacobian(shared_logistic!, datasets, u0, truth; solver = Tsit5())
    @test size(jacobian.jacobian) == (2 * length(times), 2)
    @test all(isfinite, jacobian.jacobian)

    fisher = fisher_information(
        shared_logistic!, datasets, u0, truth;
        parameter_names = [:r, :K], solver = Tsit5(),
    )
    @test fisher.numerical_rank == 2
    @test isfinite(fisher.condition_number)
    @test nrow(fisher.table) == 2

    global_sensitivity = global_sensitivity_analysis(
        shared_logistic!, datasets, u0;
        bounds = bounds, parameter_names = [:r, :K], n_samples = 8,
        rng = MersenneTwister(15), solver = Tsit5(),
    )
    @test nrow(global_sensitivity.samples) == 8
    @test nrow(global_sensitivity.pointwise) == 2 * length(times) * 2
    @test nrow(global_sensitivity.summary) == 2
    @test global_sensitivity.success_rate > 0.0

    profiles = profile_likelihood(
        shared_logistic!, datasets, u0, [0.25, 20.0];
        bounds = bounds, parameter_names = [:r, :K],
        values = Dict(:r => [0.2, 0.35, 0.5], :K => [15.0, 25.0, 35.0]),
        solver = Tsit5(), optimizer = :nelder_mead, maxiters = 600,
    )
    @test nrow(profiles.profile) == 6
    @test nrow(profiles.confidence_intervals) == 2
    @test all(isfinite, profiles.profile.sse)

    paired = paired_profile_likelihood(
        shared_logistic!, datasets, u0, [0.25, 20.0];
        bounds = bounds, parameter_names = [:r, :K], pair = (:r, :K),
        values = Dict(:r => [0.25, 0.35], :K => [20.0, 25.0]),
        solver = Tsit5(), optimizer = :nelder_mead, maxiters = 400,
    )
    @test nrow(paired.surface) == 4
    @test paired.region.parameter_1 == "r"

    hierarchical = hierarchical_joint_fit(
        shared_logistic!,
        [
            (name = "sensitive", dataset_specs = [datasets[1]], u0 = [u0[1]]),
            (name = "resistant", dataset_specs = [datasets[2]], u0 = [u0[2]]),
        ],
        [0.25, 20.0];
        bounds = bounds, parameter_names = [:r, :K], varying_parameters = [:r],
        random_effect_sd = 0.5, solver = Tsit5(), maxiters = 400,
    )
    @test length(hierarchical.group_params) == 2
    @test nrow(hierarchical.group_parameters) == 4
    @test isfinite(hierarchical.pooled_bic)

    boot = bootstrap_joint_fit(
        shared_logistic!, datasets, u0, [0.25, 20.0];
        bounds = bounds, n_bootstrap = 3, method = :parametric,
        rng = MersenneTwister(22), solver = Tsit5(), optimizer = :nelder_mead,
        maxiters = 500,
    )
    @test nrow(boot.replicates) == 3
    @test 0.0 <= boot.success_rate <= 1.0

    recovery = synthetic_recovery_benchmark(
        shared_logistic!, datasets, u0, truth;
        bounds = bounds, starts = [[0.25, 20.0]], n_simulations = 3,
        rng = MersenneTwister(32), solver = Tsit5(), optimizer = :nelder_mead,
        maxiters = 500,
    )
    @test nrow(recovery.replicates) == 3
    @test nrow(recovery.summary) == 2

    structural = structural_identifiability_report(map, [:r, :K]; known_inputs = [:drug_concentration])
    @test structural.status == "requires_symbolic_backend"
    completed_structural = structural_identifiability_report(
        map,
        [:r, :K];
        backend = (observation_map, parameter_names, known_inputs) -> (
            status = "globally_identifiable",
            model = observation_map.name,
            parameter_count = length(parameter_names),
            input_count = length(known_inputs),
        ),
    )
    @test completed_structural.status == "backend_completed"

    symbolic_model = StructuralIdentifiability.@ODEmodel(
        x'(t) = a * x(t),
        y(t) = x(t),
    )
    global_structural = structural_identifiability(symbolic_model; funcs_to_check = [symbolic_model.parameters[1]])
    @test nrow(global_structural.table) == 1
    @test global_structural.table.identifiability[1] == "globally"
end
