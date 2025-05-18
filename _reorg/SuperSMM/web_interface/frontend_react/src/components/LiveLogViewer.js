// web_interface/frontend_react/src/components/LiveLogViewer.js
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';

const MAX_LOG_LINES = 500;
const MAX_RETRY_ATTEMPTS = 5;
const INITIAL_RETRY_DELAY_MS = 1000; // 1 second

function LiveLogViewer() {
    const [logFiles, setLogFiles] = useState([]);
    const [selectedLogFile, setSelectedLogFile] = useState('');
    const [logMessages, setLogMessages] = useState([]);
    const [isLoadingFiles, setIsLoadingFiles] = useState(false);
    const [error, setError] = useState('');
    const [wsStatus, setWsStatus] = useState('Disconnected');
    const [initialLines, setInitialLines] = useState(50);
    const [autoScroll, setAutoScroll] = useState(true);
    const [filterText, setFilterText] = useState('');
    const [isCaseSensitive, setIsCaseSensitive] = useState(false);
    const [useRegex, setUseRegex] = useState(false);
    const [regexError, setRegexError] = useState('');
    const [retryAttempt, setRetryAttempt] = useState(0);

    const ws = useRef(null);
    const logContainerRef = useRef(null);
    const retryTimeoutRef = useRef(null); // To store setTimeout ID for retries

    useEffect(() => {
        setIsLoadingFiles(true);
        axios.get('/api/logs/files')
            .then(response => {
                const filesData = response.data; // This is an array of objects like {name: "...", ...}
                setLogFiles(filesData); // Store the full data if needed for the dropdown display
                setIsLoadingFiles(false);
                if (filesData && filesData.length > 0) {
                    // Backend sends them sorted, newest first.
                    const newestFileObject = filesData[0]; 
                    if (newestFileObject && newestFileObject.name) {
                        setSelectedLogFile(newestFileObject.name); // Set the filename string
                    }
                }
            })
            .catch(err => {
                console.error("Error fetching log files:", err);
                setError('Failed to load log files list.');
                setIsLoadingFiles(false);
            });
    }, []); // Empty dependency array means this runs once on mount


    const formatTimestamp = () => {
        const now = new Date();
        return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
    };

    const clearScheduledRetry = () => {
        if (retryTimeoutRef.current) {
            clearTimeout(retryTimeoutRef.current);
            retryTimeoutRef.current = null;
        }
    };

    const connectWebSocket = useCallback((isManualAttempt = true) => {
        if (!selectedLogFile) return;

        if (isManualAttempt) {
            setRetryAttempt(0); // Reset retries if user manually connects
            clearScheduledRetry();
        }

        if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
            ws.current.close();
        }

        setLogMessages([]); // Clear logs on new connection attempt
        setWsStatus(retryAttempt > 0 ? `Reconnecting (Attempt ${retryAttempt}/${MAX_RETRY_ATTEMPTS})...` : 'Connecting...');
        setError('');
        setRegexError('');

        const wsUrl = `ws://localhost:8008/api/logs/stream/${selectedLogFile}?lines=${initialLines}`;
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            setWsStatus('Connected');
            setRetryAttempt(0); // Reset retry attempts on successful connection
            clearScheduledRetry();
        };

        ws.current.onmessage = (event) => {
            setLogMessages(prevMessages => {
                const newMessage = { timestamp: formatTimestamp(), content: event.data };
                const newMessages = [...prevMessages, newMessage];
                return newMessages.length > MAX_LOG_LINES ? newMessages.slice(-MAX_LOG_LINES) : newMessages;
            });
        };

        ws.current.onerror = (socketError) => {
            console.error("WebSocket error:", socketError);
            // Error event will typically be followed by a close event, which handles retries.
            // Only set general error if not already in a retry sequence.
            if (retryAttempt === 0) {
                setError('WebSocket connection error.'); 
            }
            // wsStatus will be updated by onclose
        };

        ws.current.onclose = (event) => {
            clearScheduledRetry(); // Clear any existing scheduled retry, e.g. if closed manually
            if (!event.wasClean && selectedLogFile) { // only retry if not a clean close and a file is selected
                if (retryAttempt < MAX_RETRY_ATTEMPTS) {
                    const currentAttempt = retryAttempt + 1;
                    setRetryAttempt(currentAttempt);
                    const delay = INITIAL_RETRY_DELAY_MS * Math.pow(2, currentAttempt - 1);
                    setWsStatus(`Reconnecting (Attempt ${currentAttempt}/${MAX_RETRY_ATTEMPTS} in ${delay/1000}s)...`);
                    setError('Connection lost. Attempting to reconnect...'); 
                    retryTimeoutRef.current = setTimeout(() => connectWebSocket(false), delay);
                } else {
                    setWsStatus('Failed to reconnect');
                    setError(`Failed to reconnect after ${MAX_RETRY_ATTEMPTS} attempts. Please check connection or try manually.`);
                    setRetryAttempt(0); // Reset for next manual attempt
                }
            } else if (event.wasClean) {
                setWsStatus('Disconnected');
                setRetryAttempt(0); // Reset on clean disconnect
            } else { // Not clean, but no file selected, or some other edge case
                setWsStatus('Disconnected');
                setRetryAttempt(0);
            }
        };
    }, [selectedLogFile, initialLines, retryAttempt]); // Added retryAttempt to dependencies

    // New useEffect to connect when selectedLogFile is set (and not already connecting/connected)
    useEffect(() => {
        if (selectedLogFile && wsStatus === 'Disconnected' && retryAttempt === 0) {
            // Only auto-connect if a file is selected, we are truly disconnected,
            // and not in the middle of a retry cycle.
            connectWebSocket(false); // false indicates not a manual attempt
        }
    }, [selectedLogFile, connectWebSocket, wsStatus, retryAttempt]); // Dependencies
    useEffect(() => {
        if (autoScroll && logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logMessages, autoScroll]);

    // Cleanup WebSocket and any pending timeouts on component unmount
    useEffect(() => () => {
        clearScheduledRetry();
        if (ws.current) {
            ws.current.onopen = null;
            ws.current.onmessage = null;
            ws.current.onerror = null;
            ws.current.onclose = null;
            ws.current.close();
        }
    }, []);

    const handleFileSelect = (fileName) => {
        setSelectedLogFile(fileName);
        clearScheduledRetry();
        setRetryAttempt(0);
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.close(); // This will trigger onclose, but wasClean should be true or handled
        }
        // If a file is selected, a manual connection attempt is implied for the next action or if auto-connect is implemented
        // For now, user has to click connect explicitly.
        setWsStatus('Disconnected'); // Reset status, ready for manual connect
        setLogMessages([]);
    };

    const clearLogs = () => setLogMessages([]);

    const getStatusColor = () => {
        if (wsStatus === 'Connected') return '#28a745';
        if (wsStatus.startsWith('Connecting') || wsStatus.startsWith('Reconnecting')) return '#ffc107';
        if (wsStatus === 'Error' || wsStatus.startsWith('Failed') || wsStatus.startsWith('Disconnected (')) return '#dc3545';
        return '#6c757d';
    };

    const filteredLogMessages = useMemo(() => {
        setRegexError('');
        if (!filterText) return logMessages;
        try {
            let regex;
            if (useRegex) {
                regex = new RegExp(filterText, isCaseSensitive ? '' : 'i');
            }
            return logMessages.filter(msg => {
                const content = msg.content;
                if (useRegex && regex) {
                    return regex.test(content);
                }
                const source = isCaseSensitive ? content : content.toLowerCase();
                const target = isCaseSensitive ? filterText : filterText.toLowerCase();
                return source.includes(target);
            });
        } catch (e) {
            setRegexError('Invalid Regular Expression: ' + e.message);
            return logMessages;
        }
    }, [logMessages, filterText, isCaseSensitive, useRegex]);

    useEffect(() => {
        if (autoScroll && logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [filteredLogMessages, autoScroll]);
    
    const handleManualConnect = () => {
        clearScheduledRetry();
        setRetryAttempt(0);
        connectWebSocket(true); // Pass true for manual attempt
    };

    const handleManualDisconnect = () => {
        clearScheduledRetry();
        setRetryAttempt(0);
        if (ws.current) {
            ws.current.close(); // This should lead to a clean disconnect
        }
        setWsStatus('Disconnected');
    };


    if (isLoadingFiles) return <p>Loading log files list...</p>;

    const controlGroupStyle = { marginBottom: '10px', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '10px' };
    const inputStyle = { padding: '8px', borderRadius: '4px', border: '1px solid #ccc' };
    const buttonStyle = { padding: '8px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', backgroundColor: '#007bff', color: 'white' };
    const disabledButtonStyle = { ...buttonStyle, backgroundColor: '#6c757d', cursor: 'not-allowed' };

    const canConnect = selectedLogFile && (wsStatus === 'Disconnected' || wsStatus === 'Failed to reconnect' || wsStatus.startsWith('Disconnected ('));
    const canDisconnect = wsStatus === 'Connected' || wsStatus.startsWith('Connecting') || wsStatus.startsWith('Reconnecting');

    return (
        <div style={{ padding: '20px', fontFamily: 'Roboto, Arial, sans-serif' }}>
            <h2 style={{marginBottom: '15px'}}>Live Log Viewer</h2>
            {error && <p style={errorMsgStyle}>Error: {error}</p>} 
            {regexError && <p style={errorMsgStyle}>Regex Error: {regexError}</p>}

            <div style={controlGroupStyle}>
                <label htmlFor="logFileSelect">Log File:</label>
                <select id="logFileSelect" value={selectedLogFile} onChange={(e) => handleFileSelect(e.target.value)} style={{...inputStyle, flexGrow: 1, minWidth: '200px'}}>
                    <option value="">-- Select a Log --</option>
                    {logFiles.map(file => <option key={file.name} value={file.name}>{file.name} ({(file.size_bytes / 1024).toFixed(2)} KB)</option>)}
                </select>
                <label htmlFor="initialLines">Initial Lines:</label>
                <input type="number" id="initialLines" value={initialLines} onChange={(e) => setInitialLines(Math.max(1, parseInt(e.target.value, 10) || 50))} min="1" max="1000" style={{...inputStyle, width: '70px'}}/>
                <span style={{ padding: '8px 12px', borderRadius: '4px', color: 'white', backgroundColor: getStatusColor(), fontWeight: 'bold' }}>{wsStatus}</span>
            </div>

            <div style={controlGroupStyle}>
                <button onClick={handleManualConnect} style={!canConnect ? disabledButtonStyle : buttonStyle} disabled={!canConnect}>
                    {wsStatus === 'Failed to reconnect' || (ws.current && ws.current.readyState === WebSocket.CLOSED && retryAttempt > 0) ? 'Retry Connect' : (wsStatus === 'Connected' ? 'Reconnect' : 'Connect')}
                </button>
                <button onClick={handleManualDisconnect} style={!canDisconnect ? disabledButtonStyle : {...buttonStyle, backgroundColor: '#ffc107'}} disabled={!canDisconnect}>
                    Disconnect
                </button>
                <button onClick={clearLogs} style={logMessages.length === 0 ? disabledButtonStyle : {...buttonStyle, backgroundColor: '#17a2b8' }} disabled={logMessages.length === 0}>
                    Clear Logs
                </button>
                <input type="text" placeholder="Filter logs..." value={filterText} onChange={(e) => setFilterText(e.target.value)} style={{...inputStyle, flexGrow: 1, minWidth: '200px'}}/>
            </div>
            
            <div style={controlGroupStyle}>
                 <label style={{ display: 'flex', alignItems: 'center'}}>
                    <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} style={{ marginRight: '5px' }}/>Auto-scroll
                </label>
                <label style={{ display: 'flex', alignItems: 'center'}}>
                    <input type="checkbox" checked={isCaseSensitive} onChange={(e) => setIsCaseSensitive(e.target.checked)} style={{ marginRight: '5px' }}/>Case Sensitive
                </label>
                <label style={{ display: 'flex', alignItems: 'center'}}>
                    <input type="checkbox" checked={useRegex} onChange={(e) => setUseRegex(e.target.checked)} style={{ marginRight: '5px' }}/>Use Regex
                </label>
            </div>

            <pre ref={logContainerRef} style={logDisplayStyle}>
                {filteredLogMessages.length > 0 
                    ? filteredLogMessages.map((msg, index) => (
                        <div key={index} style={{borderBottom: '1px solid #2a2a2a', paddingBottom:'2px', marginBottom:'2px'}}>
                            <span style={{ color: '#6a9955', marginRight: '10px' }}>{msg.timestamp}</span>
                            <span style={{color: msg.content.toLowerCase().includes('error') ? '#f44747' : (msg.content.toLowerCase().includes('warn') ? '#f4b047' : '#9cdcfe')}}>
                                {msg.content}
                            </span>
                        </div>
                      ))
                    : <div style={{color: '#888'}}>{selectedLogFile ? (wsStatus.startsWith('Connecting') || wsStatus.startsWith('Reconnecting') || wsStatus === 'Connected' ? (filterText ? 'No matching log entries.' : 'Waiting for logs...') : (wsStatus === 'Failed to reconnect' ? 'Failed to connect. Check logs or try again.' : 'Disconnected. Select a log and connect.')) : 'Select a log file and connect.'}</div>
                }
            </pre>
        </div>
    );
}

const logDisplayStyle = {
    backgroundColor: '#1e1e1e', 
    color: '#d4d4d4', 
    fontFamily: 'Menlo, Monaco, Consolas, "Courier New", monospace',
    fontSize: '13px',
    padding: '15px',
    marginTop: '10px',
    height: '500px',
    overflowY: 'scroll',
    border: '1px solid #333',
    borderRadius: '4px',
    whiteSpace: 'pre-wrap', 
    wordBreak: 'break-all'  
};

const errorMsgStyle = {
    color: '#721c24', 
    backgroundColor: '#f8d7da', 
    border: '1px solid #f5c6cb', 
    padding: '10px', 
    borderRadius: '5px', 
    marginBottom: '10px'
};

export default LiveLogViewer;
