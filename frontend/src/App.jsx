import { useMemo, useState } from 'react';
import './App.css';
import { analyzePose, getApiBaseUrl } from './api/analyze';
import { PoseVisualizer } from './PoseVisualizer';
import { MetricsCharts } from './MetricsCharts';

const EMPTY_FRAME = Array(57).fill(0);
const DEFAULT_PAYLOAD = JSON.stringify(
  {
    pose: [EMPTY_FRAME, EMPTY_FRAME],
    transpose: false,
  },
  null,
  2,
);

function App() {
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [transpose, setTranspose] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const metricsPreview = useMemo(() => {
    if (!result?.performance_metrics) return [];
    return Object.entries(result.performance_metrics).slice(0, 6);
  }, [result]);

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      setPayload(text);
      setError('');
    } catch (err) {
      setError(`Unable to read file: ${err.message}`);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setResult(null);

    let parsedPayload;
    try {
      parsedPayload = JSON.parse(payload);
    } catch {
      setError('Input must be valid JSON.');
      return;
    }

    if (!Array.isArray(parsedPayload.pose)) {
      setError('Payload must contain a "pose" array.');
      return;
    }

    parsedPayload.transpose = transpose;
    setIsLoading(true);

    try {
      const response = await analyzePose(parsedPayload);
      setResult(response);
    } catch (err) {
      setError(err.message ?? 'Failed to analyze pose.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>3D Pose Analyzer</h1>
        <p>
          Paste pose JSON (from `pose_examples.json`, recorded sensors, etc.) and
          send it to the FastAPI backend&apos;s <code>/analyze</code> endpoint.
        </p>
        <span className="app__endpoint">
          Target API: <code>{getApiBaseUrl()}/analyze</code>
        </span>
      </header>

      <main className="app__main">
        <section className="panel panel--form">
          <form onSubmit={handleSubmit}>
            <label className="panel__label">
              Pose JSON
              <textarea
                value={payload}
                onChange={(event) => setPayload(event.target.value)}
                spellCheck={false}
                rows={12}
                placeholder='{"pose": [[...57 floats per frame...]], "transpose": false}'
              />
            </label>

            <div className="panel__controls">
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={transpose}
                  onChange={(event) => setTranspose(event.target.checked)}
                />
                Input is already shaped as [57, num_frames]
              </label>

              <label className="file-input">
                <span>Load JSON file</span>
                <input
                  type="file"
                  accept="application/json"
                  onChange={handleFileUpload}
                />
              </label>
            </div>

            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Analyzing…' : 'Analyze Pose'}
            </button>
          </form>

          {error ? <p className="status status--error">{error}</p> : null}
        </section>

        {result && (
          <section className="panel panel--visualizer">
            <h2>Visualizer</h2>
            <PoseVisualizer 
              inputPoseData={result.skeleton_nodes?.data} 
              correctedPoseData={result.target_pose?.data} 
            />
          </section>
        )}

        <section className="panel panel--results">
          <h2>Response</h2>
          {!result && !error && (
            <p className="placeholder">
              Submit a payload to see model predictions, key metrics, and raw
              outputs here.
            </p>
          )}

          {result ? (
            <>
              <div className="results__summary">
                <div>
                  <p className="results__label">Predicted class</p>
                  <p className="results__value">
                    {result.prediction?.predicted_class?.name ??
                      result.prediction?.predicted_class?.id ??
                      'Unknown'}
                  </p>
                  <p className="results__confidence">
                    Confidence:{' '}
                    {(
                      (result.prediction?.predicted_class?.confidence ?? 0) * 100
                    ).toFixed(2)}
                    %
                  </p>
                </div>
                <div>
                  <p className="results__label">Input shape</p>
                  <code>
                    {result.target_pose?.shape?.join(' × ') ?? '57 × 100'}
                  </code>
                  <p className="results__label">Skeleton shape</p>
                  <code>
                    {result.skeleton_nodes?.shape?.join(' × ') ?? '100 × 19 × 3'}
                  </code>
                </div>
              </div>

              {metricsPreview.length > 0 && (
                <div className="metrics-preview">
                  <p className="results__label">Metrics snapshot</p>
                  <dl>
                    {metricsPreview.map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>{formatMetric(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )}

              {result.performance_metrics && (
                <MetricsCharts metrics={result.performance_metrics} />
              )}

              <details className="raw-json" open>
                <summary>Raw JSON response</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </>
          ) : null}
        </section>
      </main>
    </div>
  );
}

function formatMetric(value) {
  try {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value.toFixed(2) : String(value);
    }
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, (key, val) => {
        if (typeof val === 'number') {
          return Number.isFinite(val) ? Number(val.toFixed(2)) : String(val);
        }
        return val;
      });
    }
    return String(value);
  } catch (e) {
    return 'Error';
  }
}

export default App;
