# CLAUDE.md

Julia package for fitting growth ODE models to time-series data: model registry,
fitting, model comparison, staged/joint workflows, and identifiability analysis.

## Layout
- `src/`: package code. Biggest files are `workflow.jl`, `fitting.jl`,
  `identifiability.jl`, `analysis.jl`, `registry.jl`, `models.jl`.
- `test/runtests.jl`: the CI test suite (`julia --project -e 'using Pkg; Pkg.test()'`).
  `test/julia/` holds ad-hoc debug scripts, not part of CI.
- `docs/`: Documenter site (`docs/make.jl`); doctests run in CI.
- `context/`: spec files. Follow `context/ai-workflow-rules.md` and
  `context/code-standards.md`; don't modify `Project.toml`/`Manifest.toml`
  unless asked.

## Model delegation

Keep the main session's model for work that needs judgment. Hand mechanical,
well-scoped work to cheaper subagents with the Agent tool's `model` parameter.

**Use `model: "haiku"` for** lookups and mechanical edits where the answer is
easy to check:
- Finding files, functions, or where a symbol is used (use the `Explore` agent).
- Listing or summarizing directory contents, CSV headers, or output files.
- Renaming, reformatting, fixing typos, updating paths or docstrings.
- Running a command and reporting what it printed (tests, `Pkg.status()`, git log).

**Use `model: "sonnet"` for** contained implementation work with a clear spec:
- Writing or updating a single function, script, or test from an explicit description.
- Plotting scripts that follow an existing pattern or style guide.
- Summarizing a long notebook, log, or report into the facts you need.
- Drafting docs, READMEs, changelog entries, or PR descriptions.

**Keep on the main model** (do not delegate):
- Choosing or changing model structure: ODE right-hand sides, priors, bounds,
  identifiability, likelihoods, BIC/AIC comparisons.
- Debugging wrong numbers, failed fits, or solver instability.
- Interpreting results and deciding what they mean for the science.
- Cross-file design decisions and anything the user will act on directly.
- Reviewing and integrating whatever a subagent returns.

**How to delegate well**
- Give the subagent everything it needs in the prompt: file paths, the exact
  change, and what "done" looks like. It does not see this conversation.
- Ask for a short result (paths, line numbers, a diff summary), not a dump.
- Run independent subagents in parallel in one message.
- Verify a subagent's edits before building on them: read the diff and run the
  relevant check. If a cheap model gets it wrong twice, do it yourself.
- Don't delegate a task smaller than writing its prompt. A single grep or a
  one-line edit is faster to do directly.

**Repo-specific examples**
- Haiku: "Where is `run_staged_pipeline` called?", "Which exported symbols lack
  docstrings?", running `Pkg.test()` and reporting failures, updating
  `CHANGELOG.md` entries from a list you provide.
- Sonnet: adding a test for a function whose behavior you've already specified,
  writing a docstring or doc page for an existing API, adding a new model to
  the registry when its equations and parameters are already decided.
- Main model: changes to fitting/optimization logic, solver choices,
  identifiability methods, public API design, and debugging numerical or
  type-inference failures.
