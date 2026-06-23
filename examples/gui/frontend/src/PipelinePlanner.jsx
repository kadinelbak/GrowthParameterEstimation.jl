import { useState } from 'react'

const PipelinePlanner = ({ csvResults, availableModels, modelTemplates, stages, onChange }) => {
  const [localStages, setLocalStages] = useState(stages)

  const getModelSpec = (modelName) => {
    // First look in customModels (passed as availableModels? but we need the full object)
    // Actually, availableModels is just an array of strings (model names) from the context.
    // We don't have the full custom models object here. We'll assume that the modelTemplates
    // contains the templates and we also need to access custom models from somewhere else.
    // Since we don't have custom models in props, we'll rely on modelTemplates only.
    // But note: the availableModels prop is an array of model names (strings) that are either
    // from custom models or from template variants.
    // We'll look in modelTemplates by variant.
    for (const variant in modelTemplates) {
      if (modelTemplates[variant].variant === modelName) {
        return modelTemplates[variant]
      }
    }
    // If not found, return a default spec to avoid errors
    return {
      paramNames: [],
      defaultBounds: [],
      defaultMathTex: '',
      defaultPlainMath: '',
      equation: '',
      defaultRhs: ''
    }
  }

  const updateStage = (stageIndex, updates) => {
    setLocalStages(prev => {
      const newStages = [...prev]
      newStages[stageIndex] = { ...newStages[stageIndex], ...updates }
      return newStages
    })
    onChange(newStages)
  }

  const updateModelBounds = (stageIndex, modelName, paramIndex, boundType, value) => {
    setLocalStages(prev => {
      const newStages = [...prev]
      const stage = { ...newStages[stageIndex] }
      if (!stage.bounds) stage.bounds = {}
      if (!stage.bounds[modelName]) {
        // Initialize with default bounds from model spec
        const spec = getModelSpec(modelName)
        stage.bounds[modelName] = spec.defaultBounds.map(b => ([b[0], b[1]]))
      }
      const bounds = [...stage.bounds[modelName]]
      if (boundType === 'low') {
        bounds[paramIndex][0] = parseFloat(value) || 0
      } else {
        bounds[paramIndex][1] = parseFloat(value) || 0
      }
      stage.bounds[modelName] = bounds
      newStages[stageIndex] = stage
      return newStages
    })
    onChange(localStages)
  }

  return (
    <div>
      {localStages.map((stage, stageIndex) => (
        <div key={stage.id} className="card stageCard">
          <div className="cardHeader">
            <h3>Stage {stageIndex + 1}</h3>
            <div className="stageControls">
              <button
                type="button"
                className="moveBtn"
                disabled={stageIndex === 0}
                onClick={() => {
                  if (stageIndex > 0) {
                    setLocalStages(prev => {
                      const newStages = [...prev]
                      const temp = newStages[stageIndex]
                      newStages[stageIndex] = newStages[stageIndex - 1]
                      newStages[stageIndex - 1] = temp
                      onChange(newStages)
                      return newStages
                    })
                  }
                }}
              >
                ⇧
              </button>
              <button
                type="button"
                className="moveBtn"
                disabled={stageIndex === localStages.length - 1}
                onClick={() => {
                  if (stageIndex < localStages.length - 1) {
                    setLocalStages(prev => {
                      const newStages = [...prev]
                      const temp = newStages[stageIndex]
                      newStages[stageIndex] = newStages[stageIndex + 1]
                      newStages[stageIndex + 1] = temp
                      onChange(newStages)
                      return newStages
                    })
                  }
                }}
              >
                ⇩
              </button>
              <button
                type="button"
                className="removeBtn"
                onClick={() => {
                  setLocalStages(prev => {
                    const newStages = prev.filter((_, idx) => idx !== stageIndex)
                    onChange(newStages)
                    return newStages
                  })
                }}
              >
                Remove
              </button>
            </div>
          </div>

          <div className="stageBody">
            <div className="fieldRow">
              <label>Dataset:</label>
              <select
                value={stage.csvFileId || ''}
                onChange={(e) => updateStage(stageIndex, { csvFileId: e.target.value })}
              >
                <option value="">-- Select Dataset --</option>
                {csvResults.map(file => (
                  <option key={file.id} value={file.id}>
                    {file.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="fieldRow">
              <label>Models:</label>
              <select
                multiple
                value={stage.models || []}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions).map(o => o.value)
                  updateStage(stageIndex, { models: selected })
                }}
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>

            <div className="fieldRow">
              <label>Fixed Parameters (one per line, format: param=value):</label>
              <textarea
                value={stage.fixedParamsText || ''}
                onChange={(e) => updateStage(stageIndex, { fixedParamsText: e.target.value })}
                rows={4}
                className="fixedParamsArea"
              />
            </div>

            {stage.models && stage.models.length > 0 && (
              <div className="boundsSection">
                <h4>Parameter Bounds (per model):</div>
                {stage.models.map(modelName => {
                  const spec = getModelSpec(modelName)
                  const paramNames = spec.paramNames
                  const defaultBounds = spec.defaultBounds
                  return (
                    <div key={modelName} className="modelBounds">
                      <h5>{modelName}</h5>
                      {paramNames.length > 0 ? (
                        <table className="boundsTable">
                          <thead>
                            <tr>
                              <th>Parameter</th>
                              <th>Lower Bound</th>
                              <th>Upper Bound</th>
                            </tr>
                          </thead>
                          <tbody>
                            {paramNames.map((paramName, paramIndex) => {
                              const currentBounds = stage.bounds && stage.bounds[modelName] ?
                                stage.bounds[modelName][paramIndex] :
                                (defaultBounds[paramIndex] || [0, 10])
                              return (
                                <tr key={paramIndex}>
                                  <td>{paramName}</td>
                                  <td>
                                    <input
                                      type="number"
                                      value={currentBounds[0]}
                                      onChange={(e) => updateModelBounds(stageIndex, modelName, paramIndex, 'low', e.target.value)}
                                      className="boundInput"
                                    />
                                  </td>
                                  <td>
                                    <input
                                      type="number"
                                      value={currentBounds[1]}
                                      onChange={(e) => updateModelBounds(stageIndex, modelName, paramIndex, 'high', e.target.value)}
                                      className="boundInput"
                                    />
                                  </td>
                                </tr>
                              )}
                            )}
                          </tbody>
                        </table>
                      ) : (
                        <p className="noParams">This model has no parameters to bound.</p>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

          </div>
        </div>
      ))}
      {localStages.length === 0 && (
        <div className="emptyStage">
          <p>No stages defined. Add stages using the planner controls above.</p>
        </div
      )}
    </div>
  )
}

export default PipelinePlanner
</path>
</write_to_file>