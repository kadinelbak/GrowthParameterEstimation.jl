# GrowthParameterEstimation GUI (Separate Frontend + Julia API)

This scaffold sets up a decoupled architecture:

- `frontend/`: React + Vite client UI.
- `backend/`: Julia API server for CSV intake and schema inspection.

## Step 1 Included

The first page supports:

- Multi-CSV upload.
- Auto-detection of `time`, `count`, and `filepath/id`-style columns.
- Selection of additional numeric variables to fit/set later.
- Table preview for validation.
- Plot overlay of uploaded time-series data.

## Stage 2 Included: Custom Logistic Model Builder

The GUI now includes a second analysis stage where you can:

- Create custom modified-logistic variants (template-based).
- Set per-parameter bounds.
- Save models for later reuse.

Saved models are persisted at:

- `examples/gui/backend/data/custom_models.json`

The backend also generates a Julia registry file:

- `examples/gui/backend/data/custom_models_registry.jl`

Load these saved models into your fitter with:

```julia
using GrowthParameterEstimation
register_models_from_file!("examples/gui/backend/data/custom_models_registry.jl")
```

## Run Backend (Julia API)

```powershell
cd examples/gui/backend
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. src/server.jl
```

Backend listens on `http://127.0.0.1:8050`.

## Run Frontend (Vite)

```powershell
cd examples/gui/frontend
npm install
npm run dev
```

Frontend runs on `http://127.0.0.1:5173` and calls backend at `http://127.0.0.1:8050` by default.

To override backend URL:

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8050"
npm run dev
```

## API Contract (current)

`POST /api/csv/inspect`

Request body:

```json
{
  "files": [
    {
      "id": "file-1",
      "name": "experiment_a.csv",
      "content": "time,count,replicate\n0,100,1\n1,130,1"
    }
  ]
}
```

Response includes, per file:

- parse status (`ok`/`error`)
- row/column counts
- column stats and inferred suggestions
- preview rows
- downsampled plot points

Additional custom-model endpoints:

- `GET /api/models/templates`
- `GET /api/models/custom`
- `POST /api/models/custom`

## Suggested Next Steps

1. Add `POST /api/session/intake` to persist mappings in backend state.
2. Add upload streaming/chunking for very large CSV collections.
3. Add Page 2 for condition grouping and model-family presets.
4. Add integration from selected fit variables into `run_pipeline` config payload.
