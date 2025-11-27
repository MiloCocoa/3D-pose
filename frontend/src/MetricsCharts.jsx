import { useState, useMemo } from 'react';

const METRIC_GROUPS = [
  {
    title: 'Knee Angles (Deg)',
    key: 'knee_angles',
    series: [
      { key: 'left_knee.angles_deg', name: 'Left Knee', color: '#8884d8' },
      { key: 'right_knee.angles_deg', name: 'Right Knee', color: '#82ca9d' },
    ],
  },
  {
    title: 'Hip Angles (Deg)',
    key: 'hip_angles',
    series: [
      { key: 'left_hip.angles_deg', name: 'Left Hip', color: '#ffc658' },
      { key: 'right_hip.angles_deg', name: 'Right Hip', color: '#ff7300' },
    ],
  },
  {
    title: 'Balance (Lateral Displacement)',
    key: 'balance',
    series: [
      { key: 'lateral_displacement.values', name: 'Displacement', color: '#00C49F' },
    ],
  },
  {
    title: 'Symmetry (Knee Angle Diff)',
    key: 'symmetry',
    series: [
      { key: 'knee_angle_symmetry.differences_rad', name: 'Diff (Rad)', color: '#FF8042' },
    ],
  },
];

function getNestedValue(obj, path) {
  if (!obj) return undefined;
  return path.split('.').reduce((acc, part) => acc && acc[part], obj);
}

// Simple SVG Line Chart Component (No external dependencies)
function SimpleLineChart({ data, series, width = 600, height = 300 }) {
  if (!data || data.length === 0) return <p>No data</p>;

  const padding = { top: 20, right: 20, bottom: 30, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Calculate Min/Max
  const allValues = data.flatMap(d => series.map(s => d[s.name])).filter(v => Number.isFinite(v));
  const minVal = Math.min(...allValues, 0); // Default to 0 baseline if values are positive
  const maxVal = Math.max(...allValues, 1);
  
  // Scales
  const xScale = (index) => (index / (data.length - 1)) * chartWidth;
  const yScale = (value) => chartHeight - ((value - minVal) / (maxVal - minVal)) * chartHeight;

  // Generate Paths
  const paths = series.map(s => {
    const points = data.map((d, i) => {
      const x = xScale(i);
      const y = yScale(d[s.name]);
      return `${x},${y}`;
    }).join(' ');
    return { ...s, path: points };
  });

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
      <g transform={`translate(${padding.left}, ${padding.top})`}>
        {/* Axes */}
        <line x1={0} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#ccc" strokeWidth="1" />
        <line x1={0} y1={0} x2={0} y2={chartHeight} stroke="#ccc" strokeWidth="1" />
        
        {/* Y-Axis Labels (Approximate) */}
        {[0, 0.5, 1].map(t => {
          const val = minVal + t * (maxVal - minVal);
          const y = chartHeight - t * chartHeight;
          return (
            <text key={t} x="-10" y={y + 5} textAnchor="end" fontSize="10" fill="#666">
              {val.toFixed(1)}
            </text>
          );
        })}

        {/* Lines */}
        {paths.map(s => (
          <polyline
            key={s.name}
            points={s.path}
            fill="none"
            stroke={s.color}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
      </g>
      
      {/* Legend */}
      <g transform={`translate(${padding.left}, 0)`}>
        {series.map((s, i) => (
          <g key={s.name} transform={`translate(${i * 100}, -10)`}>
            <rect width="10" height="10" fill={s.color} />
            <text x="15" y="9" fontSize="12" fill="#333">{s.name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

export function MetricsCharts({ metrics }) {
  const [selectedGroup, setSelectedGroup] = useState(METRIC_GROUPS[0]);

  if (!metrics) return null;

  // Prepare data
  const firstSeriesKey = selectedGroup.series[0].key;
  const firstDataArray = getNestedValue(metrics[selectedGroup.key], firstSeriesKey);

  if (!firstDataArray || !Array.isArray(firstDataArray)) {
    return (
      <div className="metrics-charts">
        <div className="chart-controls">
          {METRIC_GROUPS.map((group) => (
            <button
              key={group.title}
              className={selectedGroup.title === group.title ? 'active' : ''}
              onClick={() => setSelectedGroup(group)}
            >
              {group.title}
            </button>
          ))}
        </div>
        <p style={{ padding: '1rem', color: '#64748b' }}>
          No graph data available for <strong>{selectedGroup.title}</strong>.
        </p>
      </div>
    );
  }

  const chartData = firstDataArray.map((_, index) => {
    const point = { frame: index };
    selectedGroup.series.forEach((s) => {
      const arr = getNestedValue(metrics[selectedGroup.key], s.key);
      let val = arr ? arr[index] : 0;
      if (typeof val !== 'number' || !isFinite(val)) val = 0;
      point[s.name] = val;
    });
    return point;
  });

  return (
    <div className="metrics-charts">
      <div className="chart-controls">
        {METRIC_GROUPS.map((group) => (
          <button
            key={group.title}
            className={selectedGroup.title === group.title ? 'active' : ''}
            onClick={() => setSelectedGroup(group)}
          >
            {group.title}
          </button>
        ))}
      </div>

      <div className="chart-container" style={{ width: '100%', padding: '10px 0' }}>
        <h3>{selectedGroup.title}</h3>
        <SimpleLineChart data={chartData} series={selectedGroup.series} />
      </div>
    </div>
  );
}