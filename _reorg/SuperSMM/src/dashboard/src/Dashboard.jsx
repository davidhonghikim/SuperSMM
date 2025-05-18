import React, { useEffect, useState } from 'react';
import './Dashboard.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const LOG_PATH = 'http://localhost:4001/logs/training_log.csv';
const MODEL_INFO_PATH = 'http://localhost:4001/api/model-info';

function parseCSV(csv) {
  console.log('Parsing CSV:', csv); // Debug log
  if (!csv || csv.trim() === '') {
    console.log('Empty CSV data'); // Debug log
    return []; // Handle empty or whitespace-only input
  }
  const lines = csv.trim().split('\n');
  if (lines.length === 0) {
    console.log('No lines in CSV'); // Debug log
    return []; // Handle case where no lines are present after split
  }
  const headersLine = lines[0];
  if (!headersLine || headersLine.trim() === '') {
    console.log('Empty header line'); // Debug log
    return []; // Handle empty header line
  }
  const headers = headersLine.split(',');
  console.log('CSV Headers:', headers); // Debug log
  
  const rows = lines.slice(1).map(line => {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, i) => {
      row[header.trim()] = values[i]?.trim() || ''; // Ensure value is at least an empty string
    });
    return row;
  }).filter(row => Object.keys(row).length > 0); // Filter out potentially empty rows
  
  console.log('Parsed rows:', rows); // Debug log
  return rows;
}

function formatTime(seconds) {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hrs}h ${mins}m ${secs}s`;
}

function formatBytes(bytes) {
  const sizes = ['B', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 B';
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)), 10);
  return `${(bytes / (1024 ** i)).toFixed(2)} ${sizes[i]}`;
}

export default function Dashboard() {
  const [logRows, setLogRows] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState({
    currentEpoch: 0,
    model: '',
    lastUpdated: '',
    status: 'loading'
  });
  const [stats, setStats] = useState({
    currentEpoch: 0,
    totalEpochs: 50,
    currentLoss: 0,
    bestLoss: Infinity,
    lastValidationError: null,
    lastSerPercent: null,
    timeElapsed: 0,
    estimatedTimeRemaining: 0,
    batchSize: 0,
    learningRate: 0,
    gpuMemory: 0,
    memoryUsage: 0,
    datasetSize: 0,
    validationSamples: 0,
    lastCheckpoint: '',
    trainingHistory: [],
  });

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(MODEL_INFO_PATH);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setModelInfo(data);
    } catch (e) {
      console.error('Error fetching model info:', e);
      // Don't set an error state here, just log it
    }
  };

  const fetchLog = async () => {
    try {
      console.log('Fetching log from:', LOG_PATH); // Debug log
      const res = await fetch(LOG_PATH);
      console.log('Fetch response:', res.status, res.statusText); // Debug log
      
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const text = await res.text();
      console.log('Received text length:', text.length); // Debug log
      
      const rows = parseCSV(text);
      console.log('Setting log rows:', rows); // Debug log
      setLogRows(rows);
      
      if (rows.length > 0) {
        const lastRow = rows[rows.length - 1];
        const firstRow = rows[0];
        const firstTimestamp = new Date(firstRow.timestamp);
        const lastTimestamp = new Date(lastRow.timestamp);
        const timeElapsed = (lastTimestamp - firstTimestamp) / 1000;
        
        const currentEpoch = parseInt(lastRow.epoch);
        const losses = rows.map(r => parseFloat(r.loss)).filter(l => !isNaN(l));
        const bestLoss = Math.min(...losses);
        
        // Process training history for charts
        const trainingHistory = rows.map(row => ({
          epoch: parseInt(row.epoch),
          loss: parseFloat(row.loss),
          validationError: row.validation_error ? parseFloat(row.validation_error) : null,
          serPercent: row.ser_percent ? parseFloat(row.ser_percent) : null,
          memoryUsage: parseFloat(row.memory_usage_mb),
          gpuMemory: parseFloat(row.gpu_memory_mb),
        }));

        setStats({
          currentEpoch,
          totalEpochs: parseInt(lastRow.remaining_epochs) + currentEpoch,
          currentLoss: parseFloat(lastRow.loss),
          bestLoss,
          lastValidationError: lastRow.validation_error || null,
          lastSerPercent: lastRow.ser_percent || null,
          timeElapsed,
          estimatedTimeRemaining: timeElapsed * (parseInt(lastRow.remaining_epochs) / currentEpoch),
          batchSize: parseInt(lastRow.batch_size),
          learningRate: parseFloat(lastRow.learning_rate),
          gpuMemory: parseFloat(lastRow.gpu_memory_mb),
          memoryUsage: parseFloat(lastRow.memory_usage_mb),
          datasetSize: parseInt(lastRow.dataset_size),
          validationSamples: parseInt(lastRow.validation_samples) || 0,
          lastCheckpoint: lastRow.checkpoint_path,
          trainingHistory,
        });
      }
      
      setLastUpdate(new Date().toLocaleTimeString());
      setError(null);
    } catch (e) {
      console.error('Error fetching log:', e); // Debug log
      setError(`Error fetching log: ${e.message}`);
    }
  };

  useEffect(() => {
    console.log('Dashboard mounted, setting up fetch intervals'); // Debug log
    fetchLog();
    fetchModelInfo();
    
    const logInterval = setInterval(fetchLog, 2000); // Refresh every 2 seconds
    const modelInfoInterval = setInterval(fetchModelInfo, 5000); // Refresh model info less frequently
    
    return () => {
      clearInterval(logInterval);
      clearInterval(modelInfoInterval);
    }
  }, []);

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>OMR Training Dashboard</h1>
        <div className="dashboard-controls">
          <button className="refresh-button" onClick={() => { fetchLog(); fetchModelInfo(); }}>Refresh Now</button>
          <div className="dashboard-meta">
            <span>Last update: {lastUpdate || 'Never'}</span>
            {error && <span className="dashboard-error">{error}</span>}
          </div>
        </div>
      </div>
      
      <div className="model-status-card">
        <h3>Model Status: {modelInfo.model || 'Unknown'}</h3>
        <div className={`status-indicator ${modelInfo.status}`}></div>
        <p>Status: <strong>{modelInfo.status === 'completed' ? 'Training Completed' : 'Training in Progress'}</strong></p>
        <p>Current Epoch: {modelInfo.currentEpoch}</p>
        <p>Last Updated: {modelInfo.lastUpdated ? new Date(modelInfo.lastUpdated).toLocaleString() : 'Unknown'}</p>
      </div>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Progress</h3>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{width: `${(stats.currentEpoch / stats.totalEpochs) * 100}%`}}
            />
          </div>
          <p>Epoch {stats.currentEpoch} of {stats.totalEpochs}</p>
          <p>Dataset Size: {stats.datasetSize.toLocaleString()} samples</p>
          <p>Batch Size: {stats.batchSize}</p>
          <p>Learning Rate: {stats.learningRate}</p>
        </div>
        
        <div className="stat-card">
          <h3>Loss</h3>
          <p>Current: {stats.currentLoss.toFixed(4)}</p>
          <p>Best: {stats.bestLoss.toFixed(4)}</p>
          <p>Validation Error: {stats.lastValidationError || 'N/A'}</p>
          <p>SER: {stats.lastSerPercent ? `${stats.lastSerPercent}%` : 'N/A'}</p>
        </div>
        
        <div className="stat-card">
          <h3>Resources</h3>
          <p>Memory Usage: {formatBytes(stats.memoryUsage * 1024 * 1024)}</p>
          <p>GPU Memory: {formatBytes(stats.gpuMemory * 1024 * 1024)}</p>
          <p>Validation Samples: {stats.validationSamples}</p>
          <p>Last Checkpoint: {stats.lastCheckpoint || 'None'}</p>
        </div>
        
        <div className="stat-card">
          <h3>Time</h3>
          <p>Elapsed: {formatTime(stats.timeElapsed)}</p>
          <p>Remaining: {formatTime(stats.estimatedTimeRemaining)}</p>
          <p>Samples/sec: {((stats.batchSize * stats.currentEpoch) / stats.timeElapsed).toFixed(2)}</p>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h3>Training Progress</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={stats.trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Loss" />
              <Line type="monotone" dataKey="validationError" stroke="#82ca9d" name="Validation Error" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Symbol Error Rate</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={stats.trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="serPercent" stroke="#ff7300" name="SER %" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Resource Usage</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={stats.trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="memoryUsage" stroke="#8884d8" name="Memory (MB)" />
              <Line type="monotone" dataKey="gpuMemory" stroke="#82ca9d" name="GPU Memory (MB)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="table-container">
        <h3>Training Log</h3>
        {logRows.length > 0 ? (
          <table className="dashboard-table">
            <thead>
              <tr>
                {Object.keys(logRows[0]).map((k) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logRows.slice(-10).map((row, i) => (
                <tr key={i}>
                  {Object.values(row).map((v, j) => (
                    <td key={j}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="no-data">
            <p>No training progress yet.</p>
            <p className="debug-info">Last error: {error || 'None'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
