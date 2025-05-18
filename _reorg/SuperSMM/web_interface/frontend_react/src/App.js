// web_interface/frontend_react/src/App.js
import React from 'react';
import './App.css';
import DocExplorer from './components/DocExplorer';
import DebugSymbolExplorer from './components/DebugSymbolExplorer';
import LiveLogViewer from './components/LiveLogViewer'; // Import the new component

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>SuperSMM Project Interface</h1>
      </header>
      <main className="App-main">
        <section style={{ marginBottom: '30px', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
          <h2>Live Log Viewer</h2>
          <LiveLogViewer />
        </section>

        <hr />
        <section style={{ marginBottom: '30px', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
          <h2>Documentation Explorer</h2>
          <DocExplorer />
        </section>
        
        <hr />
        <section style={{ marginBottom: '30px', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
          <h2>Debug Symbol Explorer</h2>
          <DebugSymbolExplorer />
        </section>
      </main>
      <footer className="App-footer">
        <p>&copy; {new Date().getFullYear()} SuperSMM Project</p>
      </footer>
    </div>
  );
}

export default App;
