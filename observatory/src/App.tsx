import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { getMetrics, getPolicyEvals } from './data_loader'
// CSS for map viewer
const MAP_VIEWER_CSS = `
.map-viewer {
    position: relative;
    width: 1000px;
    margin: 20px auto;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: #f9f9f9;
    min-height: 300px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.map-viewer-title {
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
    font-size: 18px;
}
.map-viewer-img {
    max-width: 100%;
    max-height: 350px;
    display: block;
    margin: 0 auto;
}
.map-viewer-placeholder {
    text-align: center;
    color: #666;
    padding: 50px 0;
    font-style: italic;
}
.map-viewer-controls {
    display: flex;
    justify-content: center;
    margin-top: 15px;
    gap: 10px;
}
.map-button {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    font-size: 14px;
}
.map-button svg {
    width: 14px;
    height: 14px;
}
.map-button.locked {
    background: #f0f0f0;
    border-color: #aaa;
}
.map-button:hover {
    background: #f0f0f0;
}
.map-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
`;

// Types
interface MatrixRow {
  policy_uri: string
  eval_name: string
  value: number
  replay_url?: string
}

// Load data from API endpoints

function App() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [matrix, setMatrix] = useState<MatrixRow[]>([])
  const [metrics, setMetrics] = useState<string[]>([])
  const [selectedMetric, setSelectedMetric] = useState<string>("reward")
  const [selectedEval, setSelectedEval] = useState<string | null>(null)
  const [selectedReplayUrl, setSelectedReplayUrl] = useState<string | null>(null)
  const [isViewLocked, setIsViewLocked] = useState(false)
  const [isMouseOverMap, setIsMouseOverMap] = useState(false)
  const [isMouseOverHeatmap, setIsMouseOverHeatmap] = useState(false)
  
  // Map image URL helper
  const getShortName = (evalName: string) => {
    if (evalName === 'Overall') return evalName;
    return evalName.split('/').pop() || evalName;
  };

  const getMapImageUrl = (evalName: string) => {
    if (evalName.toLowerCase() === 'overall') return '';
    const shortName = getShortName(evalName);
    return `https://softmax-public.s3.amazonaws.com/policydash/evals/img/${shortName.toLowerCase()}.png`;
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, matrixData] = await Promise.all([
          getMetrics(),
          getPolicyEvals(selectedMetric)
        ]);
        setMetrics(metricsData);
        setMatrix(matrixData);
        setLoading(false);
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedMetric]);
  
  if (loading) {
    return <div>Loading data...</div>
  }
  
  if (error) {
    return <div style={{ color: 'red' }}>Error: {error}</div>
  }
  
  // Convert to heatmap format
  const policies = [...new Set(matrix.map(r => r.policy_uri))]
  const envs = [...new Set(matrix.map(r => r.eval_name))]
  const shortNames = envs.map(getShortName);
  
  // Sort envs and shortNames together based on shortNames
  const sortedIndices = shortNames
    .map((name, index) => ({ name, index }))
    .sort((a, b) => a.name.localeCompare(b.name))
    .map(item => item.index);
  
  const sortedEnvs = sortedIndices.map(i => envs[i]);
  const sortedShortNames = sortedIndices.map(i => shortNames[i]);
  
  const z = policies.map(policy => 
    sortedEnvs.map(env => {
      const row = matrix.find(r => r.policy_uri === policy && r.eval_name === env)
      return row ? row.value : 0
    })
  )

  // Map viewer functions
  const handleHeatmapHover = (event: any) => {
    if (event.points && event.points[0]) {
      const shortName = event.points[0].x;
      const policyUri = event.points[0].y;
      // Find the full eval name that corresponds to this short name
      const fullEvalName = envs.find(env => getShortName(env) === shortName);
      if (fullEvalName && !isViewLocked && shortName.toLowerCase() !== 'overall') {
        const row = matrix.find(r => r.policy_uri === policyUri && r.eval_name === fullEvalName);
        setSelectedEval(fullEvalName);
        setSelectedReplayUrl(row?.replay_url || null);
      }
    }
  };

  const handleHeatmapLeave = () => {
    setIsMouseOverHeatmap(false);
    if (!isViewLocked && !isMouseOverMap) {
      setTimeout(() => {
        if (!isMouseOverHeatmap && !isMouseOverMap) {
          setSelectedEval(null);
          setSelectedReplayUrl(null);
        }
      }, 100);
    }
  };

  const handleHeatmapEnter = () => {
    setIsMouseOverHeatmap(true);
  };

  const toggleLock = () => {
    setIsViewLocked(!isViewLocked);
  };

  const handleMapEnter = () => {
    setIsMouseOverMap(true);
  };

  const handleMapLeave = () => {
    setIsMouseOverMap(false);
    if (!isViewLocked) {
      setTimeout(() => {
        if (!isMouseOverHeatmap && !isMouseOverMap) {
          setSelectedEval(null);
          setSelectedReplayUrl(null);
        }
      }, 100);
    }
  };

  const handleReplayClick = () => {
    const replay_url_prefix = "https://metta-ai.github.io/metta/?replayUrl="
    const replay_url = selectedReplayUrl ? replay_url_prefix + selectedReplayUrl : null
    if (replay_url) {
      window.open(replay_url, '_blank');
    }
  };

  const data: Plotly.Data = {
    z,
    x: sortedShortNames,
    y: policies,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: {
      title: {
        text: selectedMetric
      }
    }
  }
  
  return (
    <div style={{ 
      fontFamily: 'Arial, sans-serif',
      margin: 0,
      padding: '20px',
      background: '#f8f9fa'
    }}>
      <style>{MAP_VIEWER_CSS}</style>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        background: '#fff',
        padding: '20px',
        borderRadius: '5px',
        boxShadow: '0 2px 4px rgba(0,0,0,.1)'
      }}>
        <h1 style={{
          color: '#333',
          borderBottom: '1px solid #ddd',
          paddingBottom: '10px',
          marginBottom: '20px'
        }}>
          Policy Evaluation Dashboard
        </h1>
        
        <div onMouseEnter={handleHeatmapEnter} onMouseLeave={handleHeatmapLeave}>
          <Plot
            data={[data]}
            layout={{
              title: {
                text: `Policy Evaluation Report: ${selectedMetric}`,
                font: {
                  size: 24
                }
              },
              height: 600,
              width: 1000,
              margin: { t: 50, b: 150, l: 200, r: 50 },
              xaxis: {
                tickangle: -45
              },
              yaxis: {
                tickangle: 0,
                automargin: true
              }
            }}
            style={{
              margin: '0 auto',
              display: 'block'
            }}
            onHover={handleHeatmapHover}
          />
        </div>

        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          marginTop: '20px',
          marginBottom: '30px',
          gap: '12px'
        }}>
          <div style={{ color: '#666', fontSize: '14px' }}>Heatmap Metric</div>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            style={{
              padding: '8px 12px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '14px',
              minWidth: '200px',
              backgroundColor: '#fff',
              cursor: 'pointer'
            }}
          >
            {metrics.map(metric => (
              <option key={metric} value={metric}>
                {metric}
              </option>
            ))}
          </select>
        </div>

        {/* Map Viewer */}
        <div 
          className="map-viewer" 
          onMouseEnter={handleMapEnter}
          onMouseLeave={handleMapLeave}
        >
          <div className="map-viewer-title">
            {selectedEval || 'Map Viewer'}
          </div>
          {!selectedEval ? (
            <div className="map-viewer-placeholder">
              Hover over an evaluation name or cell to see the environment map
            </div>
          ) : (
            <img 
              className="map-viewer-img" 
              src={getMapImageUrl(selectedEval)}
              alt={`Environment map for ${selectedEval}`}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const placeholder = target.parentElement?.querySelector('.map-viewer-placeholder') as HTMLDivElement;
                if (placeholder) {
                  placeholder.textContent = `No map available for ${selectedEval}`;
                  placeholder.style.display = 'block';
                }
              }}
            />
          )}
          
          <div className="map-viewer-controls">
            <button 
              className={`map-button ${isViewLocked ? 'locked' : ''}`}
              onClick={toggleLock}
              title="Lock current view (or click cell)"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
              </svg>
              <span>{isViewLocked ? 'Unlock View' : 'Lock View'}</span>
            </button>
            <button 
              className={`map-button ${!selectedReplayUrl ? 'disabled' : ''}`}
              onClick={handleReplayClick}
              title="Open replay in Mettascope"
              disabled={!selectedReplayUrl}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z" clipRule="evenodd" />
                <path fillRule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.194a.75.75 0 00-.053 1.06z" clipRule="evenodd" />
              </svg>
              <span>Open Replay</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App 