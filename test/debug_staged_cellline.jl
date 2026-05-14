using GrowthParameterEstimation
using CSV
using DataFrames

df = CSV.read("examples/gui/data/cellline_stages.csv", DataFrame)
cfg = default_config(output_dir="results/gui_staged_debug")

run = run_staged_pipeline(
    df;
    stages=default_stages(),
    config=cfg,
    selection_mode=:best_bic,
    strict_schema=false,
    qc_before_fit=true,
    preflight_before_fit=true,
    export_stage_results=true,
)

println("COMPLETED=", run.completed)
println("HALTED=", run.halted_stage)
println("FAIL_ROWS=", nrow(run.failures))
println("--- FAILURES ---")
show(stdout, MIME("text/plain"), run.failures)
println()

println("--- STAGES ---")
for s in run.stages
    println("stage=", s.name, " status=", s.status, " n_conditions=", s.n_conditions, " selected=", s.selected_model)
    println("candidates=", join(s.candidate_models, ","))
end
