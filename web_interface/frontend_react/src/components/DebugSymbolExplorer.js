// web_interface/frontend_react/src/components/DebugSymbolExplorer.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function DebugSymbolExplorer() {
    const [symbols, setSymbols] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');

    useEffect(() => {
        setIsLoading(true);
        axios.get('/api/debug/symbols') // Proxied to http://localhost:8008/debug/symbols
            .then(response => {
                setSymbols(response.data);
                setIsLoading(false);
            })
            .catch(err => {
                console.error("Error fetching debug symbols:", err);
                setError('Failed to load debug symbols.');
                setIsLoading(false);
            });
    }, []);

    const filteredSymbols = symbols.filter(symbol => 
        symbol.code.toLowerCase().includes(searchTerm.toLowerCase()) ||
        symbol.message_template.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (isLoading) {
        return <p>Loading debug symbols...</p>;
    }

    if (error) {
        return <p style={{ color: 'red' }}>{error}</p>;
    }

    return (
        <div style={{ padding: '20px' }}>
            <h2>Debug Symbol Explorer</h2>
            <input 
                type="text" 
                placeholder="Search by code or message..." 
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{ marginBottom: '20px', padding: '8px', width: '300px' }}
            />
            {filteredSymbols.length === 0 && <p>No debug symbols found{searchTerm ? ' matching your search' : ''}.</p>}
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                    <tr>
                        <th style={tableHeaderStyle}>Code</th>
                        <th style={tableHeaderStyle}>Message Template</th>
                        <th style={tableHeaderStyle}>Default Color</th>
                    </tr>
                </thead>
                <tbody>
                    {filteredSymbols.map(symbol => (
                        <tr key={symbol.code}>
                            <td style={tableCellStyle}>{symbol.code}</td>
                            <td style={tableCellStyle}>{symbol.message_template}</td>
                            <td style={tableCellStyle}>{symbol.default_color || 'N/A'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

const tableHeaderStyle = {
    border: '1px solid #ccc',
    padding: '8px',
    textAlign: 'left',
    backgroundColor: '#f0f0f0'
};

const tableCellStyle = {
    border: '1px solid #ccc',
    padding: '8px',
    textAlign: 'left'
};

export default DebugSymbolExplorer;
