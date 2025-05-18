// Simple Express.js proxy server for serving logs/training_log.csv to the frontend
// Usage: node LogProxy.js

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 4001;

// Enable CORS for React app
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['GET', 'HEAD', 'OPTIONS'],
  credentials: true
}));

// Serve training log file
app.get('/logs/training_log.csv', (req, res) => {
  // Define possible paths for the log file, in order of preference
  const logPaths = [
    path.join(__dirname, '..', 'tf-deep-omr', 'logs', 'training_log.csv'),  // Primary location
    path.join(__dirname, '..', '..', 'logs', 'training_log.csv'),           // Alternate location
    path.join(__dirname, '..', '..', 'logs', 'training_run.csv')            // Legacy location
  ];
  
  // Try each path until we find a valid file
  let content = null;
  let foundPath = null;
  
  for (const logPath of logPaths) {
    try {
      if (fs.existsSync(logPath)) {
        const fileContent = fs.readFileSync(logPath, 'utf8');
        // Verify file has more than just a header line
        if (fileContent.trim().split('\n').length > 1) {
          content = fileContent;
          foundPath = logPath;
          console.log(`Using log file: ${logPath}`);
          break;
        } else {
          console.log(`Found empty log file at: ${logPath}`);
        }
      }
    } catch (error) {
      console.error(`Error checking log file at ${logPath}: ${error.message}`);
    }
  }
  
  if (content) {
    res.type('text/csv').send(content);
  } else {
    console.error('No valid log file found in any of the expected locations');
    res.status(404).send('No valid log file found');
  }
});

// Serve model information
app.get('/api/model-info', (req, res) => {
  const modelInfoPath = path.join(__dirname, '..', 'tf-deep-omr', 'model', 'primus_model', 'training_state.txt');
  
  try {
    if (fs.existsSync(modelInfoPath)) {
      const epochsCompleted = parseInt(fs.readFileSync(modelInfoPath, 'utf8').trim());
      res.json({
        currentEpoch: epochsCompleted,
        model: 'primus_model',
        lastUpdated: new Date(fs.statSync(modelInfoPath).mtime).toISOString(),
        status: 'completed'
      });
    } else {
      res.json({
        currentEpoch: 0,
        model: 'unknown',
        status: 'unknown'
      });
    }
  } catch (error) {
    console.error(`Error reading model info: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Log proxy server running on port ${PORT}`);
  console.log(`Checking for log files at the following locations:`);
  console.log(`  - ${path.join(__dirname, '..', 'tf-deep-omr', 'logs', 'training_log.csv')}`);
  console.log(`  - ${path.join(__dirname, '..', '..', 'logs', 'training_log.csv')}`);
  console.log(`  - ${path.join(__dirname, '..', '..', 'logs', 'training_run.csv')}`);
});
