import { useEffect, useMemo, useRef, useState } from 'react'
import { BlockMath } from 'react-katex'
import {
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import 'katex/dist/katex.min.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8050'
const STAGES = [
  { id: 'data', label: 'Stage 1: Data Intake' },
  { id: 'models', label: 'Stage 2: Custom Models' },
  { id: 'planner', label: 'Stage 3: Pipeline Planner' },
  { id: 'review', label: 'Stage 4: Pipeline Review' }
]

const BUILTIN_VARIANT_LABELS = {
  logistic_basic: 'Logistic Growth',
  logistic_linear_kill: 'Logistic + Linear Kill',
  theta_logistic_hill_inhibition: 'Theta Logistic + Hill Inhibition',
  theta_logistic_hill_kill: 'Theta Logistic + Hill Kill'
}

function buildUploadPayload(fileRecords) {
  return {
    files: fileRecords.map((item) => ({
      id: item.id,
      name: item.file.name,
      content: item.content
    }))
  }
}

function parseFixedParams(text) {
  const result = {}
  for (const line of String(text ?? '').split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const eqIdx = trimmed.indexOf('=')
    if (eqIdx === -1) continue
    const key = trimmed.slice(0, eqIdx).trim()
    const val = trimmed.slice(eqIdx + 1).trim()
    if (key) result[key] = val
  }
  return result
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
    plainMathText: template?.defaultPlainMath ?? template?.equation ?? '',
    rhsEquation: template?.defaultRhs ?? '',
    paramNames,
    bounds
  }
}

function replaceUntilStable(text, replacer) {
  let current = text
  while (true) {
    const next = replacer(current)
    if (next === current) {
      return next
    }
    current = next
  }
}

function convertPlainMathToLatex(input) {
  let expr = String(input ?? '').trim()
  if (!expr) {
    return ''
  }

  expr = expr
    .replace(/<=/g, ' \\leq ')
    .replace(/>=/g, ' \\geq ')
    .replace(/!=/g, ' \\neq ')
    .replace(/\*/g, ' \\cdot ')

  expr = replaceUntilStable(expr, (text) => text.replace(/frac\(([^()]+),([^()]+)\)/g, (_, num, den) => `\\frac{${num.trim()}}{${den.trim()}}`))
  expr = replaceUntilStable(expr, (text) => text.replace(/pow\(([^()]+),([^()]+)\)/g, (_, base, exp) => `{${base.trim()}}^{${exp.trim()}}`))
  expr = replaceUntilStable(expr, (text) => text.replace(/sqrt\(([^()]+)\)/g, (_, arg) => `\\sqrt{${arg.trim()}}`))
  expr = replaceUntilStable(expr, (text) => text.replace(/sub\(([^()]+),([^()]+)\)/g, (_, base, sub) => `{${base.trim()}}_{${sub.trim()}}`))
  expr = replaceUntilStable(expr, (text) => text.replace(/d\(([^()]+),([^()]+)\)/g, (_, y, x) => `\\frac{d ${y.trim()}}{d ${x.trim()}}`))
  expr = replaceUntilStable(expr, (text) => text.replace(/pd\(([^()]+),([^()]+)\)/g, (_, y, x) => `\\frac{\\partial ${y.trim()}}{\\partial ${x.trim()}}`))

  expr = expr.replace(/\b([A-Za-z][A-Za-z0-9]*)\^([A-Za-z0-9]+)\b/g, '{$1}^{$2}')
  expr = expr.replace(/\b([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)\b/g, '{$1}_{$2}')

  return expr.replace(/\s+/g, ' ').trim()
}

const GREEK_TO_NAME = {
  '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma', '\\delta': 'delta',
  '\\epsilon': 'epsilon', '\\zeta': 'zeta', '\\eta': 'eta', '\\theta': 'theta',
  '\\kappa': 'kappa', '\\lambda': 'lambda', '\\mu': 'mu', '\\nu': 'nu',
  '\\pi': 'pi', '\\rho': 'rho', '\\sigma': 'sigma', '\\tau': 'tau',
  '\\phi': 'phi', '\\omega': 'omega',
  '\\Gamma': 'Gamma', '\\Delta': 'Delta', '\\Theta': 'Theta', '\\Lambda': 'Lambda',
  '\\Pi': 'Pi', '\\Sigma': 'Sigma', '\\Phi': 'Phi', '\\Omega': 'Omega'
}

function convertPlainMathToJulia(input) {
  let expr = String(input ?? '').trim()
  if (!expr) return ''

  // Take only the right side of the first = sign
  const eqIdx = expr.indexOf('=')
  if (eqIdx !== -1) {
    expr = expr.slice(eqIdx + 1).trim()
  }

  // Replace LaTeX operators before stripping backslash words
  expr = expr.replace(/\\cdot\s*/g, '*')
  expr = expr.replace(/\\times\s*/g, '*')
  expr = expr.replace(/\\div\s*/g, '/')
  expr = expr.replace(/\\leq\s*/g, '<=')
  expr = expr.replace(/\\geq\s*/g, '>=')
  expr = expr.replace(/\\pm\s*/g, '+')
  expr = expr.replace(/\\infty\s*/g, 'Inf')

  // Replace Greek letters with plain names
  for (const [latex, name] of Object.entries(GREEK_TO_NAME)) {
    expr = expr.split(latex + ' ').join(name)
    expr = expr.split(latex).join(name)
  }

  // Convert helper functions to Julia operators
  expr = replaceUntilStable(expr, (text) =>
    text.replace(/frac\(([^()]+),([^()]+)\)/g, (_, num, den) => `(${num.trim()})/(${den.trim()})`)
  )
  expr = replaceUntilStable(expr, (text) =>
    text.replace(/pow\(([^()]+),([^()]+)\)/g, (_, base, exp) => `(${base.trim()})^(${exp.trim()})`)
  )
  expr = replaceUntilStable(expr, (text) =>
    text.replace(/sqrt\(([^()]+)\)/g, (_, arg) => `sqrt(${arg.trim()})`)
  )
  // sub(a,b) → a_b  (e.g. sub(k,kill) → k_kill)
  expr = replaceUntilStable(expr, (text) =>
    text.replace(/sub\(([^()]+),([^()]+)\)/g, (_, base, s) => `${base.trim()}_${s.trim()}`)
  )
  // d() / pd() in the RHS position — strip the wrapper, keep the numerator variable
  expr = replaceUntilStable(expr, (text) =>
    text.replace(/p?d\(([^()]+),([^()]+)\)/g, (_, y) => y.trim())
  )

  // Drop any remaining backslash sequences
  expr = expr.replace(/\\[a-zA-Z]+\s*/g, '')

  // Drop any non-Julia characters (Unicode symbols, etc.)
  expr = expr.replace(/[^\w\s+\-*/^().,]/g, '')

  return expr.replace(/\s+/g, ' ').trim()
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

function eligibleCovariateColumnsForFile(fileResult, mapping) {
  const blocked = new Set([mapping?.time, mapping?.count, mapping?.id].filter(Boolean))
  return (fileResult.columns ?? [])
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
  const eligibleCovariateColumns = eligibleCovariateColumnsForFile(fileResult, mapping)
  const covariateSet = new Set((mapping.extraColumns ?? []).filter((name) => eligibleCovariateColumns.includes(name)))

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
            <p className="fitTitle">Additional covariate columns</p>
            <p className="fieldHint">Select any numeric columns to carry forward as covariates (e.g. drug concentration, dose level). These will be available when configuring fits in later stages.</p>
            {eligibleCovariateColumns.length === 0
              ? <p className="fieldHint"><em>No additional numeric columns detected beyond time, count, and ID.</em></p>
              : (
                <div className="fitColumns">
                  {eligibleCovariateColumns.map((name) => (
                    <label key={`${fileResult.id}-${name}`} className="checkboxLabel">
                      <input
                        type="checkbox"
                        checked={covariateSet.has(name)}
                        onChange={(e) => {
                          const next = new Set(mapping.extraColumns)
                          if (e.target.checked) {
                            next.add(name)
                          } else {
                            next.delete(name)
                          }
                          onUpdate(fileResult.id, { extraColumns: Array.from(next) })
                        }}
                      />
                      <span>{name}</span>
                    </label>
                  ))}
                </div>
              )
            }
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

const MATH_KEYBOARD_TABS = [
  {
    id: 'basic',
    label: 'Basic',
    keys: [
      { label: '+', insert: ' + ', description: 'Add' },
      { label: '−', insert: ' - ', description: 'Subtract' },
      { label: '×', insert: ' * ', description: 'Multiply' },
      { label: '÷', insert: ' / ', description: 'Divide' },
      { label: '(', insert: '(', description: 'Open parenthesis' },
      { label: ')', insert: ')', description: 'Close parenthesis' },
      { label: '=', insert: ' = ', description: 'Equals' },
      { label: '^', insert: '^', description: 'Exponent (e.g. N^2)' },
      { label: '±', insert: ' \\pm ', description: 'Plus or minus' },
      { label: '∞', insert: ' \\infty ', description: 'Infinity' },
      { label: '≤', insert: ' \\leq ', description: 'Less than or equal' },
      { label: '≥', insert: ' \\geq ', description: 'Greater than or equal' },
      { label: '≠', insert: ' \\neq ', description: 'Not equal' }
    ]
  },
  {
    id: 'structures',
    label: 'Structures',
    keys: [
      { label: 'a/b', insert: 'frac( , )', description: 'Fraction — fill in the top and bottom numbers' },
      { label: 'xⁿ', insert: 'pow( , )', description: 'Power — fill in the base and exponent' },
      { label: 'x²', insert: 'pow( ,2)', description: 'Square a value' },
      { label: 'x³', insert: 'pow( ,3)', description: 'Cube a value' },
      { label: '√x', insert: 'sqrt( )', description: 'Square root' },
      { label: 'xₙ', insert: 'sub( , )', description: 'Subscript — e.g. "K half" → sub(K,half)' },
      { label: '|x|', insert: 'abs( )', description: 'Absolute value' }
    ]
  },
  {
    id: 'calculus',
    label: 'Calculus',
    keys: [
      { label: 'dN/dt', insert: 'd(N,t)', description: 'Derivative of N with respect to time t' },
      { label: 'd( )/dt', insert: 'd( ,t)', description: 'Derivative with respect to t — fill in the variable' },
      { label: 'd/d( )', insert: 'd( , )', description: 'General derivative — fill in both variables' },
      { label: '∂N/∂t', insert: 'pd(N,t)', description: 'Partial derivative of N with respect to t' },
      { label: '∂/∂( )', insert: 'pd( , )', description: 'Partial derivative — fill in both variables' }
    ]
  },
  {
    id: 'functions',
    label: 'Functions',
    keys: [
      { label: 'exp', insert: 'exp( )', description: 'Exponential: e raised to a power' },
      { label: 'log', insert: 'log( )', description: 'Natural logarithm' },
      { label: 'sin', insert: 'sin( )', description: 'Sine function' },
      { label: 'cos', insert: 'cos( )', description: 'Cosine function' },
      { label: 'tan', insert: 'tan( )', description: 'Tangent function' },
      { label: 'tanh', insert: 'tanh( )', description: 'Hyperbolic tangent — common in Hill-type inhibition equations' },
      { label: 'max', insert: 'max( , )', description: 'Larger of two values' },
      { label: 'min', insert: 'min( , )', description: 'Smaller of two values' }
    ]
  },
  {
    id: 'greek',
    label: 'Greek',
    keys: [
      { label: 'α', insert: '\\alpha ', description: 'alpha' },
      { label: 'β', insert: '\\beta ', description: 'beta' },
      { label: 'γ', insert: '\\gamma ', description: 'gamma' },
      { label: 'δ', insert: '\\delta ', description: 'delta' },
      { label: 'ε', insert: '\\epsilon ', description: 'epsilon' },
      { label: 'ζ', insert: '\\zeta ', description: 'zeta' },
      { label: 'η', insert: '\\eta ', description: 'eta' },
      { label: 'θ', insert: '\\theta ', description: 'theta' },
      { label: 'κ', insert: '\\kappa ', description: 'kappa' },
      { label: 'λ', insert: '\\lambda ', description: 'lambda' },
      { label: 'μ', insert: '\\mu ', description: 'mu' },
      { label: 'ν', insert: '\\nu ', description: 'nu' },
      { label: 'π', insert: '\\pi ', description: 'pi' },
      { label: 'ρ', insert: '\\rho ', description: 'rho' },
      { label: 'σ', insert: '\\sigma ', description: 'sigma' },
      { label: 'τ', insert: '\\tau ', description: 'tau' },
      { label: 'φ', insert: '\\phi ', description: 'phi' },
      { label: 'ω', insert: '\\omega ', description: 'omega' },
      { label: 'Γ', insert: '\\Gamma ', description: 'Gamma' },
      { label: 'Δ', insert: '\\Delta ', description: 'Delta' },
      { label: 'Θ', insert: '\\Theta ', description: 'Theta' },
      { label: 'Λ', insert: '\\Lambda ', description: 'Lambda' },
      { label: 'Π', insert: '\\Pi ', description: 'Pi (capital)' },
      { label: 'Σ', insert: '\\Sigma ', description: 'Sigma (capital)' },
      { label: 'Φ', insert: '\\Phi ', description: 'Phi (capital)' },
      { label: 'Ω', insert: '\\Omega ', description: 'Omega (capital)' }
    ]
  }
]

function CustomModelBuilder({ templates, models, modelLoadInfo, onSaved, onRefresh }) {
  const templateKeys = Object.keys(templates)
  const defaultVariant = templateKeys[0] ?? 'logistic_linear_kill'
  const [form, setForm] = useState(makeModelForm(templates, defaultVariant))
  const [mathKeyboardTab, setMathKeyboardTab] = useState('basic')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [registryFile, setRegistryFile] = useState('')
  const plainMathInputRef = useRef(null)

  useEffect(() => {
    const hasCurrent = form.variant && templates[form.variant]
    if (!hasCurrent && templateKeys.length > 0) {
      setForm(makeModelForm(templates, templateKeys[0]))
    }
  }, [templates])

  const activeTemplate = templates[form.variant]
  const plainMathPreview = useMemo(() => convertPlainMathToLatex(form.plainMathText), [form.plainMathText])
  const juliaRhs = useMemo(() => convertPlainMathToJulia(form.plainMathText), [form.plainMathText])

  function insertPlainMathHelper(snippet) {
    const el = plainMathInputRef.current
    if (!el) {
      setForm((prev) => ({ ...prev, plainMathText: `${prev.plainMathText}${snippet}` }))
      return
    }

    const start = el.selectionStart ?? form.plainMathText.length
    const end = el.selectionEnd ?? form.plainMathText.length
    setForm((prev) => {
      const current = prev.plainMathText ?? ''
      return {
        ...prev,
        plainMathText: `${current.slice(0, start)}${snippet}${current.slice(end)}`
      }
    })

    requestAnimationFrame(() => {
      const cursor = start + snippet.length
      if (plainMathInputRef.current) {
        plainMathInputRef.current.focus()
        plainMathInputRef.current.setSelectionRange(cursor, cursor)
      }
    })
  }

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
          mathText: plainMathPreview,
          rhsEquation: juliaRhs,
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

  const activeKeyboardTab = MATH_KEYBOARD_TABS.find((t) => t.id === mathKeyboardTab)

  return (
    <section className="layoutGrid modelLayout">
      <div className="leftCol">
        <section className="card">
          <div className="cardHeader">
            <h2>Loaded Model State</h2>
            <button type="button" className="secondaryBtn" onClick={onRefresh}>Refresh Loaded Models</button>
          </div>
          <p className="metaLine">Models currently loaded: {models.length}</p>
        </section>

        <section className="card">
          <div className="cardHeader">
            <h2>Build a Custom Growth Model</h2>
          </div>
          <p className="subtext">
            Design your own mathematical model and save it for use in the fitter.
            Give it a name, write your equation using the math keyboard below, then set your parameters.
          </p>

          <div className="modelFormGrid">
            <label>
              <span>Model Name</span>
              <input
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g. Logistic with Drug Kill"
              />
            </label>
            <label>
              <span>Starting Template</span>
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
            <span>Notes (optional)</span>
            <textarea
              value={form.description}
              onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
              rows={2}
              placeholder="Describe what makes this model different from the base template..."
            />
          </label>

          <div className="sectionDivider">
            <h3>Your Equation (Visual Display)</h3>
            <p className="subtext">
              Type your equation in the box below and use the math keyboard to insert symbols, fractions, Greek letters, and more.
              The preview updates live — this is exactly how it will look in a textbook or paper.
            </p>
          </div>

          <div className="equationPreviewBig">
            {form.plainMathText.trim()
              ? <MathTextBlock text={plainMathPreview} fallback="(could not render — check your equation text)" />
              : <p className="equationPlaceholder">Your equation will appear here as you type...</p>
            }
          </div>

          <label>
            <span>Type your equation here</span>
            <textarea
              ref={plainMathInputRef}
              value={form.plainMathText}
              onChange={(e) => setForm((prev) => ({ ...prev, plainMathText: e.target.value }))}
              rows={3}
              placeholder="Example: d(N,t) = r * N * (1 - frac(N,K)) - sub(k,kill) * dose * N"
            />
          </label>

          {activeTemplate?.equation && (
            <p className="fieldHint">Template starting point: <span style={{ fontFamily: 'monospace' }}>{activeTemplate.equation}</span></p>
          )}

          {juliaRhs && (
            <p className="fieldHint">Computation: <code style={{ background: '#e8f0e8', padding: '0.1rem 0.35rem', borderRadius: 4, fontFamily: 'monospace' }}>{juliaRhs}</code></p>
          )}

          <div className="mathKeyboard">
            <div className="mathKeyboardTabs">
              {MATH_KEYBOARD_TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`mkTab ${mathKeyboardTab === tab.id ? 'active' : ''}`}
                  onClick={() => setMathKeyboardTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="mathKeyboardKeys">
              {activeKeyboardTab?.keys.map((key, i) => (
                <button
                  key={i}
                  type="button"
                  className="mkKey"
                  title={key.description}
                  onClick={() => insertPlainMathHelper(key.insert)}
                >
                  {key.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sectionDivider">
            <h3>Parameters</h3>
            <p className="subtext">
              These are the unknown constants the fitter will estimate from your data.
              Give each one a short name and a plausible search range (min to max).
            </p>
          </div>

          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Min</th>
                  <th>Max</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(form.paramNames ?? []).map((param, idx) => (
                  <tr key={idx}>
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
              + Add Parameter
            </button>
            <button type="button" onClick={saveModel} disabled={saving || !form.name.trim() || !form.variant || !juliaRhs.trim()}>
              {saving ? 'Saving...' : 'Save Model'}
            </button>
          </div>

          {success && <p className="successLine">{success}</p>}
          {error && <p className="error">{error}</p>}
          {registryFile && <p className="metaLine">Registry file: {registryFile}</p>}
        </section>
      </div>

      <aside className="rightCol">
        <section className="card">
          <h2>Saved Models</h2>
          <p className="subtext">These are persisted and can be loaded into the fitter later.</p>
        </section>

        {models.map((model) => (
          <section className="card" key={model.name}>
            <div className="cardHeader">
              <h3>{model.name}</h3>
              <span className="variantBadge">{model.variant}</span>
            </div>
            {model.description && <p className="subtext">{model.description}</p>}
            <div className="equationPreview">
              <MathTextBlock text={model.mathText} fallback="No equation display saved" />
            </div>

            <p className="metaLine">Parameters: {(model.paramNames ?? []).join(', ')}</p>
            <p className="metaLine">Saved: {model.createdAt}</p>
          </section>
        ))}
      </aside>
    </section>
  )
}

function makePipelineStage(index) {
  return {
    id: `stage-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    label: `Stage ${index + 1}`,
    description: '',
    csvFileId: null,
    models: [],
    fixedParamsText: '',
    carryFromStageId: null  // id of a prior pipeline stage whose fitted params are locked in for this stage
  }
}

function PipelinePlanner({ csvResults, availableModels, modelTemplates, stages, onChange }) {
  const allModels = useMemo(() => {
    const builtIn = Object.entries(BUILTIN_VARIANT_LABELS).map(([variant, name]) => ({
      name,
      paramNames: modelTemplates?.[variant]?.paramNames ?? []
    }))
    const customNames = new Set(builtIn.map((m) => m.name))
    const customs = availableModels
      .filter((m) => !customNames.has(m.name))
      .map((m) => ({ name: m.name, paramNames: m.paramNames ?? [] }))
    return [...builtIn, ...customs]
  }, [modelTemplates, availableModels])

  function addStage() {
    onChange([...stages, makePipelineStage(stages.length)])
  }

  function removeStage(id) {
    onChange(stages.filter((s) => s.id !== id))
  }

  function updateStage(id, patch) {
    onChange(stages.map((s) => s.id === id ? { ...s, ...patch } : s))
  }

  function moveStage(id, dir) {
    const idx = stages.findIndex((s) => s.id === id)
    if (idx === -1) return
    const next = stages.slice()
    const swap = idx + dir
    if (swap < 0 || swap >= next.length) return
    ;[next[idx], next[swap]] = [next[swap], next[idx]]
    onChange(next)
  }

  function toggleModel(stageId, modelName) {
    const stage = stages.find((s) => s.id === stageId)
    if (!stage) return
    const has = stage.models.includes(modelName)
    updateStage(stageId, {
      models: has ? stage.models.filter((m) => m !== modelName) : [...stage.models, modelName]
    })
  }

  return (
    <section className="plannerLayout">
      <div className="plannerLeft">
        <section className="card">
          <div className="cardHeader">
            <h2>Pipeline Stages</h2>
            <button type="button" onClick={addStage}>+ Add Stage</button>
          </div>
          <p className="subtext">
            Define each analysis stage in order. Assign a dataset and pick which models to test against it.
            This is planning only — nothing runs until Stage 4.
          </p>
        </section>

        {stages.length === 0 && (
          <section className="card">
            <p className="subtext" style={{ textAlign: 'center', padding: '1rem 0' }}>
              No stages yet. Click <strong>+ Add Stage</strong> to begin planning your pipeline.
            </p>
          </section>
        )}

        {stages.map((stage, idx) => (
          <section className="card plannerStageCard" key={stage.id}>
            <div className="cardHeader">
              <div className="stageNumberBadge">{idx + 1}</div>
              <input
                className="stageLabelInput"
                value={stage.label}
                onChange={(e) => updateStage(stage.id, { label: e.target.value })}
                placeholder="Stage name..."
              />
              <div className="stageCardActions">
                <button type="button" className="iconBtn" title="Move up" onClick={() => moveStage(stage.id, -1)} disabled={idx === 0}>↑</button>
                <button type="button" className="iconBtn" title="Move down" onClick={() => moveStage(stage.id, 1)} disabled={idx === stages.length - 1}>↓</button>
                <button type="button" className="dangerBtn" onClick={() => removeStage(stage.id)}>Remove</button>
              </div>
            </div>

            <label>
              <span>Notes (optional)</span>
              <textarea
                rows={2}
                value={stage.description}
                onChange={(e) => updateStage(stage.id, { description: e.target.value })}
                placeholder="What is this stage testing or comparing?"
              />
            </label>

            <label>
              <span>Dataset</span>
              {csvResults.length === 0
                ? <p className="fieldHint">No datasets loaded yet — go to Stage 1 to upload CSV files.</p>
                : (
                  <select
                    value={stage.csvFileId ?? ''}
                    onChange={(e) => updateStage(stage.id, { csvFileId: e.target.value || null })}
                  >
                    <option value="">(none selected)</option>
                    {csvResults.filter((f) => f.ok).map((f) => (
                      <option key={f.id} value={f.id}>{f.name}</option>
                    ))}
                  </select>
                )
              }
            </label>

            {(() => {
              // Prior stages (only stages that come before this one in the list)
              const priorStages = stages.slice(0, idx)
              if (priorStages.length === 0) return null

              // Which params does the carry-source stage expose?
              const sourceStage = stages.find((s) => s.id === stage.carryFromStageId) ?? null
              const sourceParams = sourceStage
                ? [...new Set(sourceStage.models.flatMap((name) => allModels.find((m) => m.name === name)?.paramNames ?? []))]
                : []

              return (
                <div className="carrySection">
                  <label style={{ marginBottom: 0 }}>
                    <span>Carry fitted values from</span>
                    <select
                      value={stage.carryFromStageId ?? ''}
                      onChange={(e) => updateStage(stage.id, { carryFromStageId: e.target.value || null })}
                    >
                      <option value="">(none)</option>
                      {priorStages.map((s, i) => (
                        <option key={s.id} value={s.id}>{s.label || `Stage ${i + 1}`}</option>
                      ))}
                    </select>
                  </label>
                  {sourceStage && (
                    <p className="fieldHint" style={{ marginTop: '0.3rem' }}>
                      Fitted values of{' '}
                      {sourceParams.length > 0
                        ? sourceParams.map((p) => <code key={p} style={{ marginRight: '0.25rem' }}>{p}</code>)
                        : <em>no shared parameters</em>}
                      {' '}from <strong>{sourceStage.label}</strong> will be locked in as fixed inputs for this stage.
                      Override individual values below using <em>Fixed parameters</em> if needed.
                    </p>
                  )}
                </div>
              )
            })()}

            <div className="sectionDivider">
              <h3>Models to test</h3>
            </div>

            <div className="modelCheckGrid">
              {allModels.map(({ name, paramNames }) => {
                const fixedParams = parseFixedParams(stage.fixedParamsText)
                const sourceStage = stages.find((s) => s.id === stage.carryFromStageId) ?? null
                const carriedParams = sourceStage
                  ? new Set(sourceStage.models.flatMap((mn) => allModels.find((m) => m.name === mn)?.paramNames ?? []))
                  : new Set()
                const isChecked = stage.models.includes(name)
                return (
                  <label key={name} className={`modelCard${isChecked ? ' selected' : ''}`}>
                    <div className="modelCardHeader">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleModel(stage.id, name)}
                      />
                      <span className="modelCardName">{name}</span>
                    </div>
                    {paramNames.length > 0 && (
                      <div className="paramChips">
                        {paramNames.map((p) =>
                          fixedParams[p] !== undefined
                            ? <span key={p} className="paramChip fixed" title="Manually fixed — not estimated">{p} = {fixedParams[p]}</span>
                            : carriedParams.has(p)
                              ? <span key={p} className="paramChip carry" title="Locked in from a prior pipeline stage">↳ {p}</span>
                              : <span key={p} className="paramChip fitted" title="Will be estimated from data">{p}</span>
                        )}
                      </div>
                    )}
                  </label>
                )
              })}
            </div>

            {stage.models.length === 0 && (
              <p className="fieldHint">No models selected — check at least one model to test.</p>
            )}

            <div className="fixedParamsSection">
              <details>
                <summary>Fixed parameters (optional)</summary>
                <p className="subtext" style={{ marginBottom: '0.4rem' }}>
                  Enter one <code>param = value</code> per line. Fixed parameters are held constant and not estimated.
                  They apply to all selected models that share that parameter name.
                </p>
                <textarea
                  rows={3}
                  value={stage.fixedParamsText ?? ''}
                  onChange={(e) => updateStage(stage.id, { fixedParamsText: e.target.value })}
                  placeholder={'r = 2.5\nK = 1e6'}
                  style={{ fontFamily: 'monospace', fontSize: '0.83rem' }}
                />
              </details>
            </div>
          </section>
        ))}
      </div>

      <aside className="plannerRight">
        <section className="card">
          <h2>Pipeline Summary</h2>
          <p className="subtext">A quick overview of your planned pipeline.</p>
        </section>

        {stages.length === 0 && (
          <section className="card">
            <p className="subtext">No stages planned yet.</p>
          </section>
        )}

        {stages.map((stage, idx) => {
          const csvFile = csvResults.find((f) => f.id === stage.csvFileId)
          return (
            <section className="card summaryCard" key={stage.id}>
              <div className="cardHeader">
                <h3><span className="stageNumberBadge sm">{idx + 1}</span> {stage.label || `Stage ${idx + 1}`}</h3>
              </div>
              <p className="metaLine">Dataset: {csvFile ? csvFile.name : <em>not assigned</em>}</p>
              <p className="metaLine">Models: {stage.models.length === 0 ? <em>none selected</em> : stage.models.join(', ')}</p>
              {(() => {
                const fixed = parseFixedParams(stage.fixedParamsText)
                const keys = Object.keys(fixed)
                return keys.length > 0
                  ? <p className="metaLine">Fixed: {keys.map((k) => `${k}=${fixed[k]}`).join(', ')}</p>
                  : null
              })()}
              {stage.description && <p className="subtext">{stage.description}</p>}
            </section>
          )
        })}
      </aside>
    </section>
  )
}

function PipelineReview({ csvResults, stages, availableModels, modelTemplates, stageResults, runningStageId, onRunStage }) {
  const [activeIdx, setActiveIdx] = useState(0)
  const stage = stages[activeIdx] ?? null
  const csvFile = stage ? csvResults.find((f) => f.id === stage.csvFileId) : null

  const modelParamMap = useMemo(() => {
    const map = {}
    for (const [variant, name] of Object.entries(BUILTIN_VARIANT_LABELS)) {
      map[name] = modelTemplates?.[variant]?.paramNames ?? []
    }
    for (const m of (availableModels ?? [])) {
      map[m.name] = m.paramNames ?? []
    }
    return map
  }, [modelTemplates, availableModels])

  // A stage is runnable when all prior stages are done and this stage is configured
  function isRunnable(idx) {
    const s = stages[idx]
    if (!s?.csvFileId || s.models.length === 0) return false
    if (runningStageId) return false
    for (let i = 0; i < idx; i++) {
      if (stageResults[stages[i]?.id]?.status !== 'done') return false
    }
    return stageResults[s.id]?.status !== 'running'
  }

  if (stages.length === 0) {
    return (
      <section className="card">
        <h2>Stage 4: Pipeline Review</h2>
        <p className="subtext">No pipeline stages have been planned yet. Go to Stage 3 to build your pipeline first.</p>
      </section>
    )
  }

  return (
    <section className="reviewLayout">
      <aside className="reviewNav">
        <section className="card">
          <h2>Pipeline Stages</h2>
          <p className="subtext">{stages.length} stage{stages.length !== 1 ? 's' : ''} planned</p>
        </section>

        {stages.map((s, idx) => {
          const complete = s.csvFileId && s.models.length > 0
          const sr = stageResults[s.id]
          const statusBadge = sr?.status === 'done' ? '✓' : sr?.status === 'error' ? '!' : sr?.status === 'running' ? '…' : null
          return (
            <button
              key={s.id}
              type="button"
              className={`reviewNavBtn ${activeIdx === idx ? 'active' : ''} ${complete ? '' : 'incomplete'}`}
              onClick={() => setActiveIdx(idx)}
            >
              <span className="stageNumberBadge sm">{idx + 1}</span>
              <span className="reviewNavLabel">{s.label || `Stage ${idx + 1}`}</span>
              {statusBadge && <span className={`stageStatusBadge ${sr.status}`}>{statusBadge}</span>}
              {!complete && !statusBadge && <span className="incompleteTag">Needs setup</span>}
            </button>
          )
        })}
      </aside>

      <div className="reviewMain">
        {stage && (
          <>
            <section className="card">
              <div className="cardHeader">
                <h2><span className="stageNumberBadge">{activeIdx + 1}</span> {stage.label || `Stage ${activeIdx + 1}`}</h2>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button type="button" className="secondaryBtn" onClick={() => setActiveIdx(Math.max(0, activeIdx - 1))} disabled={activeIdx === 0}>← Previous</button>
                  <button type="button" className="secondaryBtn" onClick={() => setActiveIdx(Math.min(stages.length - 1, activeIdx + 1))} disabled={activeIdx === stages.length - 1}>Next →</button>
                </div>
              </div>
              {stage.description && <p className="subtext">{stage.description}</p>}
            </section>

            <section className="card">
              <h3>Dataset</h3>
              {csvFile
                ? (
                  <div className="reviewDetail">
                    <p className="reviewDetailRow"><span>File:</span> <strong>{csvFile.name}</strong></p>
                    <p className="reviewDetailRow"><span>Rows:</span> {csvFile.nRows}</p>
                    {csvFile.columns && (
                      <p className="reviewDetailRow"><span>Columns:</span> {csvFile.columns.map((c) => c.name).join(', ')}</p>
                    )}
                  </div>
                )
                : <p className="fieldHint incompleteHint">No dataset assigned — go back to Stage 3 to assign one.</p>
              }
            </section>

            <section className="card">
              <h3>Models to Test</h3>
              {stage.models.length === 0
                ? <p className="fieldHint incompleteHint">No models selected — go back to Stage 3 to pick models.</p>
                : (() => {
                  const fixedParams = parseFixedParams(stage.fixedParamsText)
                  return (
                    <div className="reviewModelList">
                      {stage.models.map((name, i) => {
                        const paramNames = modelParamMap[name] ?? []
                        return (
                          <div key={name} className="reviewModelRow">
                            <span className="reviewModelNum">{i + 1}</span>
                            <div>
                              <span>{name}</span>
                              {paramNames.length > 0 && (
                                <div className="paramChips" style={{ marginTop: '0.2rem' }}>
                                  {paramNames.map((p) =>
                                    fixedParams[p] !== undefined
                                      ? <span key={p} className="paramChip fixed" title="Fixed — not estimated">{p} = {fixedParams[p]}</span>
                                      : <span key={p} className="paramChip fitted" title="Fitted from data">{p}</span>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )
                })()
              }
            </section>

            {(() => {
              const sourceStage = stages.find((s) => s.id === stage.carryFromStageId) ?? null
              if (!sourceStage) return null
              const sourceIdx = stages.findIndex((s) => s.id === stage.carryFromStageId)
              const reviewModelParamMapLocal = { ...modelParamMap }
              const sourceCarriedParams = [...new Set(
                sourceStage.models.flatMap((mn) => reviewModelParamMapLocal[mn] ?? [])
              )]
              const manualFixed = parseFixedParams(stage.fixedParamsText)
              const effectivelyCarried = sourceCarriedParams.filter((p) => manualFixed[p] === undefined)

              // If source stage has results, show resolved values
              const sourceResult = stageResults[sourceStage.id]
              const resolvedCarry = sourceResult?.bestModel && sourceResult?.fits
                ? (() => {
                  const bestFit = sourceResult.fits.find((f) => f.model === sourceResult.bestModel)
                  return Object.fromEntries((bestFit?.params ?? []).filter((p) => effectivelyCarried.includes(p.name)).map((p) => [p.name, p.value]))
                })()
                : {}

              return (
                <section className="card">
                  <h3>Parameter Carry-Forward</h3>
                  <p className="reviewDetailRow">
                    <span>Source stage:</span>
                    <strong><span className="stageNumberBadge sm" style={{ display: 'inline-flex', verticalAlign: 'middle', marginRight: '0.3rem' }}>{sourceIdx + 1}</span>{sourceStage.label || `Stage ${sourceIdx + 1}`}</strong>
                  </p>
                  {effectivelyCarried.length > 0
                    ? (
                      <div className="paramChips" style={{ marginTop: '0.45rem' }}>
                        {effectivelyCarried.map((p) => (
                          resolvedCarry[p] !== undefined
                            ? <span key={p} className="paramChip carry" title="Resolved from prior stage fit">↳ {p} = {resolvedCarry[p].toPrecision(4)}</span>
                            : <span key={p} className="paramChip carry" title="Will be locked in from prior stage results">↳ {p}</span>
                        ))}
                      </div>
                    )
                    : <p className="fieldHint">No parameters overlap between the source and this stage's models.</p>
                  }
                  {Object.keys(manualFixed).length > 0 && (
                    <p className="subtext" style={{ marginTop: '0.4rem' }}>
                      Manual overrides in Fixed parameters will take priority over carried values.
                    </p>
                  )}
                </section>
              )
            })()}

            {/* ── Run button + results ───────────────────────────────── */}
            {(() => {
              const sr = stageResults[stage.id]
              const canRun = isRunnable(activeIdx)
              const isRunning = sr?.status === 'running'
              const isDone = sr?.status === 'done'
              const isError = sr?.status === 'error'
              const configured = stage.csvFileId && stage.models.length > 0
              const priorBlocked = activeIdx > 0 && stageResults[stages[activeIdx - 1]?.id]?.status !== 'done'

              return (
                <section className="card runCard">
                  <div className="cardHeader">
                    <h3>Run Stage {activeIdx + 1}</h3>
                    <button
                      type="button"
                      className="runBtn"
                      disabled={!canRun}
                      onClick={() => onRunStage(activeIdx)}
                    >
                      {isRunning ? 'Running…' : isDone ? '↺ Re-run' : '▶ Run Stage'}
                    </button>
                  </div>
                  {!configured && <p className="fieldHint incompleteHint">Assign a dataset and select models in Stage 3 before running.</p>}
                  {configured && priorBlocked && <p className="fieldHint">Complete the prior pipeline stage first.</p>}
                  {isRunning && <p className="subtext runningHint">Fitting models — this may take a moment…</p>}
                  {isError && <p className="error">{sr.error}</p>}
                  {isDone && (
                    <div className="resultsBlock">
                      {/* ── Fit chart ─────────────────────────────────── */}
                      {sr.scatter?.length > 0 && (
                        <div className="fitChartWrap">
                          <ResponsiveContainer width="100%" height={240}>
                            <ComposedChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#c9d6d0" />
                              <XAxis dataKey="x" type="number" name="Time" domain={['auto', 'auto']} scale="auto" />
                              <YAxis dataKey="y" type="number" name="Count" domain={['auto', 'auto']} width={55} />
                              <Tooltip
                                formatter={(v, name) => [typeof v === 'number' ? v.toPrecision(4) : v, name]}
                                labelFormatter={(l) => `t = ${Number(l).toPrecision(4)}`}
                              />
                              <Legend />
                              <Scatter
                                name="Data"
                                data={sr.scatter.map(([t, c]) => ({ x: t, y: c }))}
                                fill="#8b9a96"
                                opacity={0.55}
                                r={3}
                              />
                              {sr.rankedFits.map((fit, ri) => {
                                const color = makeColor(ri)
                                const isBest = fit.model === sr.bestModel
                                return (
                                  <Line
                                    key={fit.model}
                                    data={(fit.curve ?? []).map(([t, c]) => ({ x: t, y: c }))}
                                    dataKey="y"
                                    name={`${fit.model}${isBest ? ' ★' : ''}`}
                                    stroke={color}
                                    strokeWidth={isBest ? 2.5 : 1.5}
                                    strokeDasharray={isBest ? undefined : '5 3'}
                                    dot={false}
                                    isAnimationActive={false}
                                  />
                                )
                              })}
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                      {/* ── Results table ─────────────────────────────── */}
                      <table className="resultsTable">
                        <thead>
                          <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>BIC</th>
                            <th>SSR</th>
                            {(sr.rankedFits[0]?.params ?? []).map((p) => <th key={p.name}>{p.name}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {sr.rankedFits.map((fit, ri) => (
                            <tr key={fit.model} className={fit.model === sr.bestModel ? 'bestRow' : ''}>
                              <td>{ri + 1}{fit.model === sr.bestModel ? ' ★' : ''}</td>
                              <td>{fit.model}</td>
                              <td>{fit.bic?.toFixed(2) ?? '—'}</td>
                              <td>{fit.ssr?.toExponential(2) ?? '—'}</td>
                              {(fit.params ?? []).map((p) => <td key={p.name}>{p.value?.toPrecision(4)}</td>)}
                            </tr>
                          ))}
                          {sr.fits.filter((f) => !f.ok).map((fit) => (
                            <tr key={fit.model} className="errorRow">
                              <td>—</td>
                              <td>{fit.model}</td>
                              <td colSpan={2 + (sr.rankedFits[0]?.params?.length ?? 0)} style={{ color: 'var(--accent-2)', fontSize: '0.82rem' }}>
                                Failed: {fit.error}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {sr.bestModel && (
                        <p className="successLine" style={{ marginTop: '0.5rem' }}>
                          Best model by BIC: <strong>{sr.bestModel}</strong>
                        </p>
                      )}
                    </div>
                  )}
                </section>
              )
            })()}
          </>
        )}
      </div>
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
  const [pipelineStages, setPipelineStages] = useState([])
  const [stageResults, setStageResults] = useState({})   // { [stageId]: { status, fits, rankedFits, bestModel, error } }
  const [runningStageId, setRunningStageId] = useState(null)

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
        const eligibleCovariateColumns = eligibleCovariateColumnsForFile(f, baseMapping)

        initialMappings[f.id] = {
          ...baseMapping,
          extraColumns: (f.columnSuggestions?.fitCandidates ?? []).filter((name) => eligibleCovariateColumns.includes(name))
        }
      }
      setMappingByFile(initialMappings)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function runPipelineStage(stageIdx) {
    const stage = pipelineStages[stageIdx]
    if (!stage || runningStageId) return

    const fileRecord = files.find((f) => f.id === stage.csvFileId)
    if (!fileRecord) return

    setRunningStageId(stage.id)
    setStageResults((prev) => ({ ...prev, [stage.id]: { status: 'running', fits: [], rankedFits: [], bestModel: null, error: null } }))

    try {
      // Build anchor params from the carry-from stage's best fit
      let anchorParams = {}
      if (stage.carryFromStageId) {
        const sourceResult = stageResults[stage.carryFromStageId]
        if (sourceResult?.bestModel && sourceResult?.fits) {
          const bestFit = sourceResult.fits.find((f) => f.model === sourceResult.bestModel)
          if (bestFit?.params) {
            for (const { name, value } of bestFit.params) {
              anchorParams[name] = value
            }
          }
        }
      }

      // Parse manual fixed params — these override carried values
      const fixedParams = {}
      for (const [k, v] of Object.entries(parseFixedParams(stage.fixedParamsText ?? ''))) {
        const num = parseFloat(v)
        if (Number.isFinite(num)) fixedParams[k] = num
      }

      const response = await fetch(`${API_BASE}/api/pipeline/run_stage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stageId: stage.id,
          csvContent: fileRecord.content,
          csvName: fileRecord.file.name,
          models: stage.models,
          anchorParams,
          fixedParams,
        })
      })

      const payload = await response.json()
      if (!response.ok || !payload.ok) throw new Error(payload.error ?? 'Stage run failed')

      setStageResults((prev) => ({
        ...prev,
        [stage.id]: {
          status: 'done',
          fits: payload.fits ?? [],
          rankedFits: payload.rankedFits ?? [],
          bestModel: payload.bestModel ?? null,
          scatter: payload.scatter ?? [],
          error: null,
        }
      }))
    } catch (e) {
      setStageResults((prev) => ({
        ...prev,
        [stage.id]: { status: 'error', fits: [], rankedFits: [], bestModel: null, error: e.message }
      }))
    } finally {
      setRunningStageId(null)
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
                mapping={mappingByFile[fileResult.id] ?? { time: null, count: null, id: null, extraColumns: [] }}
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

      {activeStage === 'planner' && (
        <PipelinePlanner
          csvResults={results}
          availableModels={customModels}
          modelTemplates={modelTemplates}
          stages={pipelineStages}
          onChange={setPipelineStages}
        />
      )}

      {activeStage === 'review' && (
        <PipelineReview
          csvResults={results}
          stages={pipelineStages}
          availableModels={customModels}
          modelTemplates={modelTemplates}
          stageResults={stageResults}
          runningStageId={runningStageId}
          onRunStage={runPipelineStage}
        />
      )}
    </main>
  )
}

export default App
