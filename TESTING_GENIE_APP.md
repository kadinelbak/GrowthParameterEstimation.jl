# Quick Start: Testing the Genie App (Stage 4)

## Goal

Validate the migrated Genie/Stipple GUI end-to-end, including load, model selection, fit/rank, and staged analysis panels.

## Prerequisites

1. Julia v1.10+
2. Workspace root: C:\Users\elbak\Desktop\GrowthParam\GrowthParameterEstimation.jl

## Steps

### 1) Stop existing listeners on GUI ports

Run in PowerShell:

Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -in 8050,8051,8052 } |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

### 2) Instantiate GUI environment

cd examples/gui
julia --project=. -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"

### 3) Launch app (preferred compatibility entrypoint)

From workspace root:

julia --project=examples/gui examples/gui/pipeline_gui_app.jl

Alternative direct entrypoint:

julia --project=examples/gui examples/gui/pipeline_gui_app_genie.jl

Expected terminal message:

GrowthParameterEstimation GUI (Genie/Stipple) ready at http://127.0.0.1:8050

### 4) Open browser

http://127.0.0.1:8050

## Manual Smoke Checklist

- Tab 1 Load Data:
  - Click Basic Pipeline Example and confirm status, preview table, preflight panel, and overview plot render.
- Tab 3 Build Models:
  - Load a template and confirm preview updates.
  - Register a model and confirm success status.
- Tab 4 Select Models:
  - Select at least 2 models and verify equation panel updates.
- Tab 5 Fit and Rank:
  - Run Fit and Rank and verify ranking table + fit plot render.
  - Run Full Pipeline and verify summary output is shown.
- Tab 6 Staged Analysis:
  - Load staged dataset from Tab 1.
  - Run staged analysis and verify stage summary output.

## Success Criteria

- App starts without Julia exceptions.
- All migrated tabs render and react to inputs.
- Fit and staged actions complete with visible output panels.

## Troubleshooting

- Port already in use:
  - Set GPE_GUI_PORT before launch, example: $env:GPE_GUI_PORT='8051'
- Slow first launch:
  - Precompilation can take multiple minutes on first run.
- Missing package errors:
  - Re-run instantiate in examples/gui and ensure --project points to examples/gui.
- Blank output panel after action:
  - Check Julia terminal for stack trace from watcher action.

## VS Code Task

Use Tasks: Run Task -> Restart GUI App.

This task already kills ports 8050-8052 and launches the compatibility entrypoint.
