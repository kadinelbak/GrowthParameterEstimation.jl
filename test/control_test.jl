using Test
using GrowthParameterEstimation
using DifferentialEquations
using ForwardDiff
using JSON

@testset "Control handoff" begin
    spec = get_model("logistic_linear_kill")
    p = [0.6, 1000.0, 0.3]   # r, K, kill_coeff

    @testset "Registered model → ControlModel" begin
        cm = controlled_model(spec, p; input_names = [:dose])
        @test cm isa ControlModel
        @test cm.n_states == 1 && cm.n_inputs == 1 && cm.n_outputs == 1
        @test cm.param_names == [:r, :K, :kill_coeff]
        @test cm.state_names == [:N]
        @test cm.input_names == [:dose]
        @test cm.metadata[:source_model] == "logistic_linear_kill"
        @test isempty(cm.param_samples)
        @test_throws ArgumentError controlled_model(spec, p[1:2])

        # control_rhs matches the registered RHS driven by the same constant exposure
        f = control_rhs(cm)
        x = [200.0]
        du = zeros(1)
        spec.ode!(du, x, p, 0.0, ConstantExposure(1.5))
        @test f(x, [1.5]) ≈ du
        @test f(x, [1.5], p, 3.0) ≈ du

        dx = zeros(1)
        control_rhs!(cm)(dx, x, [1.5], p, 0.0)
        @test dx ≈ du
    end

    @testset "MPC-convention functions" begin
        cm = controlled_model(spec, p)
        m = mpc_functions(cm)
        @test (m.nx, m.nu, m.ny) == (1, 1, 1)
        @test m.p == p
        xdot = zeros(1)
        m.f!(xdot, [200.0], [0.5], nothing, m.p)
        @test xdot ≈ control_rhs(cm)([200.0], [0.5])
        y = zeros(1)
        m.h!(y, [200.0], nothing, m.p)
        @test y == [200.0]
    end

    @testset "Discrete dynamics: accuracy and differentiability" begin
        cm = controlled_model(spec, p)
        F = discrete_dynamics(cm; dt = 2.0, substeps = 40)
        x0 = [100.0]

        # constant input over one step agrees with the package simulator
        sim = simulate(spec, [0.0, 2.0], p; u0 = x0, exposure = ConstantExposure(0.8))
        @test sim.success
        @test F(x0, [0.8])[1] ≈ sim.states[1, end] rtol = 1e-6

        # Jacobians by ForwardDiff agree with central finite differences
        Jx = ForwardDiff.jacobian(x -> F(x, [0.8]), x0)
        Ju = ForwardDiff.jacobian(u -> F(x0, u), [0.8])
        Jp = ForwardDiff.jacobian(q -> F(x0, [0.8], q), p)
        h = 1e-4
        @test Jx[1, 1] ≈ (F(x0 .+ h, [0.8])[1] - F(x0 .- h, [0.8])[1]) / 2h rtol = 1e-5
        @test Ju[1, 1] ≈ (F(x0, [0.8 + h])[1] - F(x0, [0.8 - h])[1]) / 2h rtol = 1e-5
        @test size(Jp) == (1, 3)
        @test Ju[1, 1] < 0   # more drug, fewer cells

        @test_throws ArgumentError discrete_dynamics(cm; dt = 0.0)
    end

    @testset "Piecewise-constant schedules" begin
        cm = controlled_model(spec, p)
        sched = InputSchedule([0.0, 4.0, 8.0], [0.0, 1.0, 0.0])
        @test sched(0.0) == [0.0]
        @test sched(5.0) == [1.0]
        @test sched(20.0) == [0.0]
        @test_throws ArgumentError InputSchedule([2.0, 0.0], [1.0, 0.0])
        @test_throws ArgumentError InputSchedule([0.0], [1.0, 0.0])

        times = collect(0.0:1.0:12.0)
        out = simulate_schedule(cm, [50.0], sched, times)
        @test out.success
        @test size(out.states) == (1, length(times))
        @test size(out.outputs) == (1, length(times))
        @test out.inputs[1, :] == [t >= 4 && t < 8 ? 1.0 : 0.0 for t in times]

        stepped = SteppedExposure([0.0, 4.0, 8.0], [0.0, 1.0, 0.0])
        ref = simulate(spec, times, p; u0 = [50.0], exposure = stepped)
        @test ref.success
        @test out.states[1, end] ≈ ref.states[1, end] rtol = 1e-4

        # the schedule agrees with chaining the discrete map over 2-day holds
        F = discrete_dynamics(cm; dt = 2.0, substeps = 40)
        x = [50.0]
        for u in [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
            x = F(x, [u])
        end
        @test x[1] ≈ out.states[1, end] rtol = 1e-6
    end

    @testset "Bootstrap parameter sets and ensembles" begin
        samples = [p .* s for s in (0.9, 1.0, 1.1)]
        boot = (estimates = samples, success_rate = 1.0)
        cm = controlled_model(spec, p; param_samples = boot)
        @test length(cm.param_samples) == 3
        @test length(parameter_sets(cm)) == 4
        @test parameter_sets(cm)[1] == p
        @test length(parameter_sets(cm; include_estimate = false)) == 3
        @test_throws ArgumentError controlled_model(spec, p; param_samples = [p[1:2]])
        @test_throws ArgumentError controlled_model(spec, p; param_samples = (fit = nothing,))

        sched = InputSchedule([0.0, 6.0], [0.0, 1.0])
        ens = schedule_ensemble(cm, [50.0], sched, 0.0:2.0:14.0)
        @test length(ens) == 4
        @test all(r -> r.success, ens)
        finals = [r.states[1, end] for r in ens]
        @test length(unique(round.(finals; digits = 6))) > 1
    end

    @testset "Generic RHS with two inputs and custom outputs" begin
        # two populations, input 1 = drug (kills S), input 2 = resource limitation (lowers K)
        function sr!(dx, x, p, t, exposure)
            drug, lowres = exposure(t)
            rS, rR, K, beta, c = p
            Keff = K / (1 + c * lowres)
            total = x[1] + x[2]
            dx[1] = rS * x[1] * (1 - total / Keff) - beta * drug * x[1]
            dx[2] = rR * x[2] * (1 - total / Keff)
            return nothing
        end
        q = [0.8, 0.5, 4000.0, 0.6, 0.4]
        cm = controlled_model(sr!, q; n_states = 2, n_inputs = 2,
                              state_names = [:S, :R], input_names = [:dose, :fasting],
                              param_names = [:rS, :rR, :K, :beta, :c],
                              observable = (x, p) -> [x[1] + x[2], x[2] / (x[1] + x[2])],
                              output_names = [:total, :resistant_fraction],
                              metadata = Dict(:units => "cells/image, days"))
        @test cm.n_outputs == 2
        @test cm.output_names == [:total, :resistant_fraction]
        f = control_rhs(cm)
        @test f([100.0, 100.0], [1.0, 0.0])[1] < f([100.0, 100.0], [0.0, 0.0])[1]
        @test f([100.0, 100.0], [0.0, 1.0])[2] < f([100.0, 100.0], [0.0, 0.0])[2]

        F = discrete_dynamics(cm; dt = 2.0)
        J = ForwardDiff.jacobian(u -> F([100.0, 100.0], u), [0.5, 0.5])
        @test size(J) == (2, 2)

        sched = InputSchedule([0.0, 4.0], [[0.0, 0.0], [1.0, 1.0]])
        out = simulate_schedule(cm, [100.0, 100.0], sched, 0.0:1.0:8.0)
        @test out.success
        @test size(out.inputs) == (2, 9)
        @test all(0 .<= out.outputs[2, :] .<= 1)
        @test_throws ArgumentError simulate_schedule(cm, [100.0], sched, 0.0:1.0:8.0)

        @test_throws ArgumentError controlled_model(sr!, q; n_states = 2, n_inputs = 2, input_names = [:dose])
    end

    @testset "Plain-data export" begin
        cm = controlled_model(spec, p; param_samples = [p, p .* 1.05], metadata = Dict(:stage => "Stage 2"))
        d = control_model_summary(cm)
        @test d["param_names"] == ["r", "K", "kill_coeff"]
        @test d["params"] == p
        @test length(d["param_samples"]) == 2
        @test d["metadata"]["stage"] == "Stage 2"

        path = joinpath(mktempdir(), "nested", "model.json")
        @test save_control_model(cm, path) == path
        back = JSON.parsefile(path)
        @test back["name"] == "logistic_linear_kill"
        @test Float64.(back["params"]) == p
        @test back["input_names"] == ["u"]
    end
end
