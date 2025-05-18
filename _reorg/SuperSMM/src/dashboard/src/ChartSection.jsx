import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ChartSection({ data }) {
  if (!data || data.length === 0) return <p className="no-data-message">No chart data yet.</p>;
  // Try to infer numeric columns for plotting
  const numericKeys = Object.keys(data[0] || {}).filter(
    k => k !== 'epoch' && !isNaN(parseFloat(data[0][k]))
  );
  return (
    <div style={{marginTop: '2rem'}}>
      <h2>Training Trends</h2>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          {numericKeys.map((k, i) => (
            <Line key={k} type="monotone" dataKey={k} stroke={['#8884d8','#82ca9d','#ff7300','#b00','#2d6cdf'][i%5]} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
