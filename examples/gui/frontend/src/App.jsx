import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8050'

function buildUploadPayload(fileRecords) {
  return {
    files: fileRecords.map((item) => ({
      id: item.id,
      name: item.file.name,
      content: item.content
    }))
  }
}

function makeColor(index) {
  const palette = ['#005f73', '#9b2226', '#386641', '#0a9396', '#ca6702', '#3a0ca3', '#ae2012']
  return palette[index % palette.length]
}

function FileMappingCard({ fileResult, mapping, onUpdate }) {
  const columns = fileResult.columns?.map((c) => c.name) ?? []
  const fitSet = new Set(mapping.fitCandidates)

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
              {columns.map((name) => (
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

function App() {
  const [files, setFiles] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState([])
  const [summary, setSummary] = useState(null)
  const [selectedFileId, setSelectedFileId] = useState(null)
  const [mappingByFile, setMappingByFile] = useState({})

  const selectedResult = useMemo(() => results.find((f) => f.id === selectedFileId) ?? results[0] ?? null, [results, selectedFileId])

  const chartData = useMemo(() => {
    const rows = []
    for (const item of results) {
      if (!item.ok || !item.plotSeries) {
        continue
      }
      for (const p of item.plotSeries) {
        rows.push({
          file: item.name,
          x: Number(p.x),
          y: Number(p.y)
        })
      }
    }
    return rows.sort((a, b) => a.x - b.x)
  }, [results])

  const lineFiles = useMemo(() => results.filter((f) => f.ok), [results])

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
        initialMappings[f.id] = {
          time: f.columnSuggestions?.time ?? null,
          count: f.columnSuggestions?.count ?? null,
          id: f.columnSuggestions?.id ?? null,
          fitCandidates: f.columnSuggestions?.fitCandidates ?? []
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
        <h1>Step 1: Data Intake</h1>
        <p className="subtext">
          Upload multiple CSVs, auto-detect key columns, verify table previews, and visually confirm time-series curves before fitting.
        </p>
      </header>

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
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#c9d6d0" />
                    <XAxis dataKey="x" type="number" domain={["auto", "auto"]} />
                    <YAxis dataKey="y" type="number" domain={["auto", "auto"]} />
                    <Tooltip />
                    <Legend />
                    {lineFiles.map((file, index) => (
                      <Line
                        key={file.id}
                        type="monotone"
                        dataKey="y"
                        data={chartData.filter((row) => row.file === file.name)}
                        name={file.name}
                        stroke={makeColor(index)}
                        dot={false}
                        isAnimationActive={false}
                      />
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
    </main>
  )
}

export default App
