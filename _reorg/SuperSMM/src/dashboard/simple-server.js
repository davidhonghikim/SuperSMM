const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8000;

// Define paths for log files, in order of preference
const LOG_PATHS = [
  path.join(__dirname, '..', 'tf-deep-omr', 'logs', 'training_log.csv'),  // Primary location
  path.join(__dirname, '..', '..', 'logs', 'training_log.csv'),          // Alternate location
  path.join(__dirname, '..', '..', 'logs', 'training_run.csv')           // Legacy location
];

// Create simple HTTP server
const server = http.createServer((req, res) => {
  console.log(`Request for ${req.url}`);
  
  // Handle CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.statusCode = 200;
    res.end();
    return;
  }
  
  // Serve log file
  if (req.url.includes('training_log.csv') || req.url.includes('training_run.csv')) {
    // Try each path until we find a valid file
    let content = null;
    let foundPath = null;
    
    for (const logPath of LOG_PATHS) {
      try {
        if (fs.existsSync(logPath)) {
          const fileContent = fs.readFileSync(logPath, 'utf8');
          // Check if file has actual data (more than just headers)
          if (fileContent.trim().split('\n').length > 1) {
            content = fileContent;
            foundPath = logPath;
            console.log(`Serving log file from: ${logPath}`);
            break;
          } else {
            console.log(`Found empty log file at: ${logPath}`);
          }
        }
      } catch (error) {
        console.error(`Error reading log file at ${logPath}: ${error.message}`);
      }
    }
    
    if (content) {
      res.setHeader('Content-Type', 'text/csv');
      res.statusCode = 200;
      res.end(content);
    } else {
      res.statusCode = 404;
      res.end('Log file not found or empty');
    }
    return;
  }
  
  // Serve model info
  if (req.url === '/api/model-info') {
    const modelInfoPath = path.join(__dirname, '..', 'tf-deep-omr', 'model', 'primus_model', 'training_state.txt');
    
    try {
      if (fs.existsSync(modelInfoPath)) {
        const epochsCompleted = parseInt(fs.readFileSync(modelInfoPath, 'utf8').trim());
        const modelInfo = {
          currentEpoch: epochsCompleted,
          model: 'primus_model',
          lastUpdated: new Date(fs.statSync(modelInfoPath).mtime).toISOString(),
          status: 'completed'
        };
        
        res.setHeader('Content-Type', 'application/json');
        res.statusCode = 200;
        res.end(JSON.stringify(modelInfo));
      } else {
        res.setHeader('Content-Type', 'application/json');
        res.statusCode = 200;
        res.end(JSON.stringify({
          currentEpoch: 0,
          model: 'unknown',
          status: 'unknown'
        }));
      }
    } catch (error) {
      console.error(`Error reading model info: ${error.message}`);
      res.statusCode = 500;
      res.end(JSON.stringify({ error: error.message }));
    }
    return;
  }
  
  // Serve dashboard.html (root path)
  if (req.url === '/' || req.url === '/index.html') {
    try {
      const dashboardPath = path.join(__dirname, 'dashboard.html');
      if (fs.existsSync(dashboardPath)) {
        const content = fs.readFileSync(dashboardPath, 'utf8');
        res.setHeader('Content-Type', 'text/html');
        res.statusCode = 200;
        res.end(content);
      } else {
        res.statusCode = 404;
        res.end('Dashboard file not found');
      }
    } catch (error) {
      console.error('Error reading dashboard file:', error);
      res.statusCode = 500;
      res.end('Error reading dashboard file');
    }
    return;
  }
  
  // Default 404 response
  res.statusCode = 404;
  res.end('Not Found');
});

// Start server
server.listen(PORT, () => {
  console.log(`Simple dashboard server running on http://localhost:${PORT}`);
  console.log(`Looking for log files in the following locations:`);
  LOG_PATHS.forEach(path => console.log(`  - ${path}`));
}); 