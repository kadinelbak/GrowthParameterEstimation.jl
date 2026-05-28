import { useEffect, useMemo, useState } from 'react'
import { BlockMath } from 'react-katex'
import {
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import 'katex/dist/katex.min.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8050'
const STAGES = [
  { id: 'data', label: 'Stage 1: Data Intake' },
  { id: 'models', label: 'Stage 2: Custom Models' }
]

function buildUploadPayload(fileRecords) {
  return {
    files: fileRecords.map((item) => ({
      id: item.id,
      name: item.file.name,
      content: item.content
    }))
  }
}

function makeModelForm(templateMap, variant) {
  const template = templateMap?.[variant]
  const paramNames = [...(template?.paramNames ?? [])]
  const bounds = (template?.defaultBounds ?? []).map((b) => ({ low: b[0], high: b[1] }))
  return {
    name: '',
    description: '',
    variant,
    mathText: template?.defaultMathTex ?? '',
    rhsEquation: template?.defaultRhs ?? '',
    paramNames,
    bounds
  }
}

function MathTextBlock({ text, fallback }) {
  const expr = String(text ?? '').trim()
  if (!expr) {
    return <p className="subtext">{fallback}</p>
  }
  try {
    return <BlockMath math={expr} />
  } catch {
    return <p className="metaLine">{expr}</p>
  }
}

function makeColor(index) {
  const palette = ['#005f73', '#9b2226', '#386641', '#0a9396', '#ca6702', '#3a0ca3', '#ae2012']
  return palette[index % palette.length]
}

function isNumericTypeName(typeName) {
  return /(Int|Float|UInt|BigInt|BigFloat|Rational|Decimal)/i.test(String(typeName ?? ''))
}

function isIdentifierLike(name) {
  const norm = String(name ?? '').toLowerCase()
  return norm === 'id' || norm.endsWith('_id') || /(sample|file|path|replicate|rep)\b/.test(norm)
}

function eligibleFitColumnsForFile(fileResult, mapping) {
  const blocked = new Set([mapping?.time, mapping?.count, mapping?.id].filter(Boolean))
  return (fileResult.columns ?? [])
    .filter((col) => isNumericTypeName(col.type))
    .filter((col) => !isIdentifierLike(col.name))
    .filter((col) => !blocked.has(col.name))
    .map((col) => col.name)
}

function summarizeSeries(points) {
  const grouped = new Map()

  for (const point of points) {
    const x = Number(point.x)
    const y = Number(point.y)
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      continue
    }
    const key = String(x)
    if (!grouped.has(key)) {
      grouped.set(key, { x, ys: [] })
    }
    grouped.get(key).ys.push(y)
  }

  return Array.from(grouped.values())
    .map((entry) => {
      const n = entry.ys.length
      const mean = entry.ys.reduce((acc, v) => acc + v, 0) / n
      const variance = n > 1
        ? entry.ys.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (n - 1)
        : 0

      return {
        x: entry.x,
        mean,
        sd: Math.sqrt(Math.max(0, variance)),
        n
      }
    })
    .sort((a, b) => a.x - b.x)
}

function FileMappingCard({ fileResult, mapping, onUpdate }) {
  const columns = fileResult.columns?.map((c) => c.name) ?? []
  const eligibleFitColumns = eligibleFitColumnsForFile(fileResult, mapping)
  const fitSet = new Set((mapping.fitCandidates ?? []).filter((name) => eligibleFitColumns.includes(name)))

  return (
    <section className="card">
      <div className="cardHeader">
        <h3>{fileResult.name}</h3>
        <span>{fileResult.nRows} rows</span>
      </div>

      {!fileResult.ok && <p className="error">{fileResult.error}</p>}

      {fileResult.ok && (
        <>
          <div className="mappingGrid">
            <label>
              <span>Time column</span>
              <select value={mapping.time ?? ''} onChange={(e) => onUpdate(fileResult.id, { time: e.target.value || null })}>
                <option value="">(none)</option>
                {columns.map((name) => (
                  <option value={name} key={name}>{name}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Count column</span>
              <select value={mapping.count ?? ''} onChange={(e) => onUpdate(fileResult.id, { count: e.target.value || null })}>
                <option value="">(none)</option>
                {columns.map((name) => (
                  <option value={name} key={name}>{name}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Filepath / ID</span>
              <select value={mapping.id ?? ''} onChange={(e) => onUpdate(fileResult.id, { id: e.target.value || null })}>
                <option value="">(none)</option>
                {columns.map((name) => (
                  <option value={name} key={name}>{name}</option>
                ))}
              </select>
            </label>
          </div>

          <div>
            <p className="fitTitle">Variables to fit or set later</p>
            <div className="fitColumns">
              {eligibleFitColumns.map((name) => (
                <label key={`${fileResult.id}-${name}`} className="checkboxLabel">
                  <input
                    type="checkbox"
                    checked={fitSet.has(name)}
                    onChange={(e) => {
                      const next = new Set(mapping.fitCandidates)
                      if (e.target.checked) {
                        next.add(name)
                      } else {
                        next.delete(name)
                      }
                      onUpdate(fileResult.id, { fitCandidates: Array.from(next) })
                    }}
                  />
                  <span>{name}</span>
                </label>
              ))}
            </div>
          </div>

          {fileResult.warnings?.length > 0 && (
            <ul className="warningList">
              {fileResult.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  )
}

function CustomModelBuilder({ templates, models, modelLoadInfo, onSaved, onRefresh }) {
  const templateKeys = Object.keys(templates)
  const defaultVariant = templateKeys[0] ?? 'logistic_linear_kill'
  const [form, setForm] = useState(makeModelForm(templates, defaultVariant))
  const [authoringMode, setAuthoringMode] = useState('rhs')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [registryFile, setRegistryFile] = useState('')

  useEffect(() => {
    const hasCurrent = form.variant && templates[form.variant]
    if (!hasCurrent && templateKeys.length > 0) {
      setForm(makeModelForm(templates, templateKeys[0]))
    }
  }, [templates])

  const activeTemplate = templates[form.variant]

  function updateParamName(index, value) {
    setForm((prev) => {
      const next = prev.paramNames.slice()
      next[index] = value
      return { ...prev, paramNames: next }
    })
  }

  function updateParamBounds(index, field, value) {
    setForm((prev) => {
      const next = prev.bounds.slice()
      next[index] = { ...next[index], [field]: value }
      return { ...prev, bounds: next }
    })
  }

  function addParameterRow() {
    setForm((prev) => ({
      ...prev,
      paramNames: [...prev.paramNames, `p${prev.paramNames.length + 1}`],
      bounds: [...prev.bounds, { low: 0.0, high: 1.0 }]
    }))
  }

  function removeParameterRow(index) {
    setForm((prev) => {
      const nextNames = prev.paramNames.slice()
      const nextBounds = prev.bounds.slice()
      nextNames.splice(index, 1)
      nextBounds.splice(index, 1)
      return {
        ...prev,
        paramNames: nextNames,
        bounds: nextBounds
      }
    })
  }

  async function saveModel() {
    setSaving(true)
    setError('')
    setSuccess('')

    try {
      const response = await fetch(`${API_BASE}/api/models/custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: form.name,
          description: form.description,
          variant: form.variant,
          mathText: form.mathText,
          rhsEquation: form.rhsEquation,
          paramNames: form.paramNames,
          bounds: form.bounds.map((b) => [Number(b.low), Number(b.high)])
        })
      })

      const payload = await response.json()
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error ?? 'Failed to save model')
      }

      setRegistryFile(payload.registryFile ?? '')
      setSuccess(payload.message ?? 'Model saved')
      onSaved(payload.models ?? [])
      if (onRefresh) {
        onRefresh()
      }
      setForm(makeModelForm(templates, form.variant))
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="layoutGrid modelLayout">
      <div className="leftCol">
        <section className="card">
          <div className="cardHeader">
            <h2>Loaded Model State</h2>
            <button type="button" className="secondaryBtn" onClick={onRefresh}>Refresh Loaded Models</button>
          </div>
          <p className="metaLine">Loaded models from backend: {models.length}</p>
          {modelLoadInfo?.registryFile && <p className="metaLine">Registry file: {modelLoadInfo.registryFile}</p>}
          {modelLoadInfo?.loadHint && <p className="metaLine">Load command: {modelLoadInfo.loadHint}</p>}
          {modelLoadInfo?.loadedAt && <p className="metaLine">Last refreshed: {modelLoadInfo.loadedAt}</p>}
        </section>

        <section className="card">
          <div className="cardHeader">
            <h2>Create Modified Logistic Model</h2>
          </div>
          <p className="subtext">Build a custom logistic variant, save it, and load it later in the fitter using generated Julia registry code.</p>

          <div className="modelFormGrid">
            <label>
              <span>Model Name</span>
              <input
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g. Logistic Hill"
              />
              <p className="fieldHint">Spaces are now allowed. The backend will generate a safe internal symbol automatically.</p>
            </label>
            <label>
              <span>Variant</span>
              <select
                value={form.variant}
                onChange={(e) => setForm(makeModelForm(templates, e.target.value))}
              >
                {templateKeys.map((key) => (
                  <option value={key} key={key}>{templates[key].label}</option>
                ))}
              </select>
            </label>
          </div>

          <label>
            <span>Description</span>
            <textarea
              value={form.description}
              onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
              rows={3}
              placeholder="Notes about this model variant"
            />
          </label>

          <p className="equationPreview">Template: {activeTemplate?.equation ?? 'N/A'}</p>

          <div className="authoringModeRow">
            <button
              type="button"
              className={`modeChip ${authoringMode === 'rhs' ? 'active' : ''}`}
              onClick={() => setAuthoringMode('rhs')}
            >
              Author In RHS
            </button>
            <button
              type="button"
              className={`modeChip ${authoringMode === 'math' ? 'active' : ''}`}
              onClick={() => setAuthoringMode('math')}
            >
              Author In Math Text
            </button>
          </div>

          {authoringMode === 'math' && (
            <p className="fieldHint">Math text is for readable display. The executable RHS still needs to be present below for saving into the fitter.</p>
          )}

          {authoringMode === 'rhs' && (
            <p className="fieldHint">RHS is the executable form used to generate the Julia model. Math text is optional display metadata.</p>
          )}

          {authoringMode === 'math' && (
            <>
              <label>
                <span>Math Text (LaTeX, optional)</span>
                <textarea
                  value={form.mathText}
                  onChange={(e) => setForm((prev) => ({ ...prev, mathText: e.target.value }))}
                  rows={3}
                  placeholder="Example: \\frac{dN}{dt} = rN\\left(1-\\frac{N}{K}\\right) - k_{kill}DN"
                />
              </label>

              <div className="equationPreview">
                <MathTextBlock text={form.mathText} fallback="No math text yet." />
              </div>

              <label>
                <span>Executable RHS (required for saving)</span>
                <textarea
                  value={form.rhsEquation}
                  onChange={(e) => setForm((prev) => ({ ...prev, rhsEquation: e.target.value }))}
                  rows={4}
                  placeholder="Example: r*N*(1 - N/max(K, 1e-8)) - kill_coeff*dose*N"
                />
              </label>
            </>
          )}

          {authoringMode === 'rhs' && (
            <>
              <label>
                <span>Equation RHS (sets du[1])</span>
                <textarea
                  value={form.rhsEquation}
                  onChange={(e) => setForm((prev) => ({ ...prev, rhsEquation: e.target.value }))}
                  rows={4}
                  placeholder="Example: r*N*(1 - N/max(K, 1e-8)) - kill_coeff*dose*N"
                />
              </label>

              <label>
                <span>Math Text (LaTeX, optional)</span>
                <textarea
                  value={form.mathText}
                  onChange={(e) => setForm((prev) => ({ ...prev, mathText: e.target.value }))}
                  rows={3}
                  placeholder="Example: \\frac{dN}{dt} = rN\\left(1-\\frac{N}{K}\\right) - k_{kill}DN"
                />
              </label>

              <div className="equationPreview">
                <MathTextBlock text={form.mathText} fallback="No math text yet." />
              </div>
            </>
          )}

          <p className="metaLine">Available RHS symbols: N, dose, max, min, abs, exp, log, sqrt, clamp, plus parameter names below.</p>

          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Lower Bound</th>
                  <th>Upper Bound</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {(form.paramNames ?? []).map((param, idx) => (
                  <tr key={`${idx}-${param}`}>
                    <td>
                      <input
                        type="text"
                        value={param}
                        onChange={(e) => updateParamName(idx, e.target.value)}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        value={form.bounds[idx]?.low ?? ''}
                        onChange={(e) => updateParamBounds(idx, 'low', e.target.value)}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        value={form.bounds[idx]?.high ?? ''}
                        onChange={(e) => updateParamBounds(idx, 'high', e.target.value)}
                      />
                    </td>
                    <td>
                      <button type="button" className="dangerBtn" onClick={() => removeParameterRow(idx)} disabled={form.paramNames.length <= 1}>
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="uploaderRow">
            <button type="button" className="secondaryBtn" onClick={addParameterRow}>
              Add Parameter
            </button>
            <button type="button" onClick={saveModel} disabled={saving || !form.name.trim() || !form.variant}>
              {saving ? 'Saving...' : 'Save Custom Model'}
            </button>
          </div>

          {success && <p className="successLine">{success}</p>}
          {error && <p className="error">{error}</p>}
          {registryFile && <p className="metaLine">Registry file: {registryFile}</p>}
        </section>
      </div>

      <aside className="rightCol">
        <section className="card">
          <h2>Saved Custom Models</h2>
          <p className="subtext">These are persisted for later use in the model fitter.</p>
        </section>

        {models.map((model) => (
          <section className="card" key={model.name}>
            <div className="cardHeader">
              <h3>{model.name}</h3>
              <span>{model.variant}</span>
            </div>
            {model.description && <p className="subtext">{model.description}</p>}
            <div className="equationPreview">
              <MathTextBlock text={model.mathText} fallback="No math text provided" />
            </div>
            <p className="equationPreview">du[1] = {model.rhsEquation ?? '(legacy model without stored RHS)'}</p>
            <p className="metaLine">Params: {(model.paramNames ?? []).join(', ')}</p>
            <p className="metaLine">Created: {model.createdAt}</p>
          </section>
        ))}
      </aside>
    </section>
  )
}

function App() {
  const [files, setFiles] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState([])
  const [summary, setSummary] = useState(null)
  const [selectedFileId, setSelectedFileId] = useState(null)
  const [mappingByFile, setMappingByFile] = useState({})
  const [activeStage, setActiveStage] = useState(STAGES[0].id)
  const [modelTemplates, setModelTemplates] = useState({})
  const [customModels, setCustomModels] = useState([])
  const [modelLoadInfo, setModelLoadInfo] = useState({ registryFile: '', loadHint: '', loadedAt: '' })
  const [modelsError, setModelsError] = useState('')

  const selectedResult = useMemo(() => results.find((f) => f.id === selectedFileId) ?? results[0] ?? null, [results, selectedFileId])

  const chartSeriesByFile = useMemo(() => {
    return results
      .filter((f) => f.ok)
      .map((f) => ({
        id: f.id,
        name: f.name,
        series: summarizeSeries(f.plotSeries ?? [])
      }))
      .filter((f) => f.series.length > 0)
  }, [results])

  async function loadCustomModelContext() {
    try {
      const [templatesRes, modelsRes] = await Promise.all([
        fetch(`${API_BASE}/api/models/templates`),
        fetch(`${API_BASE}/api/models/custom`)
      ])

      const templatesPayload = await templatesRes.json()
      const modelsPayload = await modelsRes.json()

      if (!templatesRes.ok || !templatesPayload.ok) {
        throw new Error(templatesPayload.error ?? 'Failed to load model templates')
      }
      if (!modelsRes.ok || !modelsPayload.ok) {
        throw new Error(modelsPayload.error ?? 'Failed to load custom models')
      }

      setModelTemplates(templatesPayload.templates ?? {})
      setCustomModels(modelsPayload.models ?? [])
      setModelLoadInfo({
        registryFile: modelsPayload.registryFile ?? '',
        loadHint: modelsPayload.loadHint ?? '',
        loadedAt: new Date().toLocaleString()
      })
      setModelsError('')
    } catch (e) {
      setModelsError(e.message)
    }
  }

  useEffect(() => {
    loadCustomModelContext()
  }, [])

  async function readFiles(fileList) {
    const next = []
    for (const file of fileList) {
      const content = await file.text()
      next.push({
        id: `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2, 9)}`,
        file,
        content
      })
    }
    return next
  }

  function updateMapping(fileId, patch) {
    setMappingByFile((prev) => ({
      ...prev,
      [fileId]: {
        ...prev[fileId],
        ...patch
      }
    }))
  }

  async function analyzeFiles() {
    if (files.length === 0) {
      setError('Select at least one CSV file first.')
      return
    }

    setBusy(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/csv/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildUploadPayload(files))
      })

      const payload = await response.json()
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error ?? 'CSV inspection failed')
      }

      setResults(payload.files ?? [])
      setSummary(payload.summary ?? null)
      setSelectedFileId(payload.files?.[0]?.id ?? null)

      const initialMappings = {}
      for (const f of payload.files ?? []) {
        const baseMapping = {
          time: f.columnSuggestions?.time ?? null,
          count: f.columnSuggestions?.count ?? null,
          id: f.columnSuggestions?.id ?? null
        }
        const eligibleFitColumns = eligibleFitColumnsForFile(f, baseMapping)

        initialMappings[f.id] = {
          ...baseMapping,
          fitCandidates: (f.columnSuggestions?.fitCandidates ?? []).filter((name) => eligibleFitColumns.includes(name))
        }
      }
      setMappingByFile(initialMappings)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="appShell">
      <header className="hero">
        <p className="eyebrow">GrowthParameterEstimation GUI</p>
        <h1>Staged Analysis Workspace</h1>
        <p className="subtext">
          Progress through stages: validate data first, then define custom model variants that can be reloaded in the fitter.
        </p>
      </header>

      <section className="stageTabs">
        {STAGES.map((stage) => (
          <button
            key={stage.id}
            type="button"
            className={`stageTab ${activeStage === stage.id ? 'active' : ''}`}
            onClick={() => setActiveStage(stage.id)}
          >
            {stage.label}
          </button>
        ))}
      </section>

      {activeStage === 'data' && (
        <>
      <section className="card uploader">
        <div className="uploaderRow">
          <input
            type="file"
            accept=".csv,text/csv"
            multiple
            onChange={async (e) => {
              const selected = await readFiles(Array.from(e.target.files ?? []))
              setFiles(selected)
            }}
          />
          <button type="button" onClick={analyzeFiles} disabled={busy || files.length === 0}>
            {busy ? 'Analyzing...' : 'Analyze CSV Files'}
          </button>
        </div>

        <p className="metaLine">
          {files.length} file(s) selected
          {summary ? ` | ${summary.successfulFiles}/${summary.fileCount} parsed | ${summary.totalRows} rows` : ''}
        </p>
        {error && <p className="error">{error}</p>}
      </section>

      {results.length > 0 && (
        <section className="layoutGrid">
          <div className="leftCol">
            <section className="card chartCard">
              <div className="cardHeader">
                <h2>Uploaded Data Plot</h2>
              </div>

              <div className="chartWrap">
                <ResponsiveContainer width="100%" height={360}>
                  <LineChart data={[]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#c9d6d0" />
                    <XAxis dataKey="x" type="number" domain={["auto", "auto"]} />
                    <YAxis type="number" domain={["auto", "auto"]} />
                    <Tooltip />
                    <Legend />
                    {chartSeriesByFile.map((file, index) => (
                      <Line
                        key={file.id}
                        type="linear"
                        dataKey="mean"
                        data={file.series}
                        name={file.name}
                        stroke={makeColor(index)}
                        dot={false}
                        isAnimationActive={false}
                      >
                        <ErrorBar dataKey="sd" width={4} strokeWidth={1} />
                      </Line>
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {selectedResult?.ok && (
              <section className="card previewCard">
                <div className="cardHeader">
                  <h2>Table Preview: {selectedResult.name}</h2>
                  <select value={selectedResult.id} onChange={(e) => setSelectedFileId(e.target.value)}>
                    {results.map((f) => (
                      <option value={f.id} key={f.id}>{f.name}</option>
                    ))}
                  </select>
                </div>

                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(selectedResult.preview?.[0] ?? {}).map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(selectedResult.preview ?? []).map((row, idx) => (
                        <tr key={idx}>
                          {Object.keys(selectedResult.preview?.[0] ?? {}).map((col) => (
                            <td key={`${idx}-${col}`}>{String(row[col] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </div>

          <aside className="rightCol">
            <section className="card">
              <h2>Column Mapping</h2>
              <p className="subtext">Review auto-detected columns and choose additional fit variables per file.</p>
            </section>

            {results.map((fileResult) => (
              <FileMappingCard
                key={fileResult.id}
                fileResult={fileResult}
                mapping={mappingByFile[fileResult.id] ?? { time: null, count: null, id: null, fitCandidates: [] }}
                onUpdate={updateMapping}
              />
            ))}
          </aside>
        </section>
      )}

      {results.length === 0 && (
        <section className="card">
          <h2>Stage 1: Data Intake</h2>
          <p className="subtext">Upload CSV files and click Analyze CSV Files to continue.</p>
        </section>
      )}
        </>
      )}

      {activeStage === 'models' && (
        <>
          {modelsError && <p className="error">{modelsError}</p>}
          <CustomModelBuilder
            templates={modelTemplates}
            models={customModels}
            modelLoadInfo={modelLoadInfo}
            onSaved={(next) => setCustomModels(next)}
            onRefresh={loadCustomModelContext}
          />
        </>
      )}
    </main>
  )
}

export default App
