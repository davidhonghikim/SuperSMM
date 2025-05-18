// web_interface/frontend_react/src/components/DocExplorer.js
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios'; // Make sure to npm install axios or yarn add axios

function DocExplorer() {
    const [references, setReferences] = useState([]);
    const [selectedDoc, setSelectedDoc] = useState(null);
    const [docContent, setDocContent] = useState({ html: '', markdown: '' });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        setIsLoading(true);
        axios.get('/api/documentation/references') // Proxied to http://localhost:8008/documentation/references
            .then(response => {
                setReferences(response.data);
                setIsLoading(false);
            })
            .catch(err => {
                console.error("Error fetching doc references:", err);
                setError('Failed to load documentation references.');
                setIsLoading(false);
            });
    }, []);

    const fetchDocContent = useCallback((path) => {
        if (!path) return;
        // The path from references_index.md might include an anchor, remove it for fetching file content
        const filePath = path.split('#')[0];
        if (!filePath) {
            setDocContent({ html: '<p>Link is an anchor only, no separate file to load.</p>', markdown: '' });
            return;
        }

        setIsLoading(true);
        axios.get(`/api/documentation/content`, { params: { path: filePath } })
            .then(response => {
                setDocContent(response.data);
                setSelectedDoc(filePath); // Keep track of which doc is shown
                setIsLoading(false);
                // If path had an anchor, try to scroll to it
                const anchor = path.split('#')[1];
                if (anchor) {
                    setTimeout(() => {
                        const element = document.getElementById(anchor);
                        if (element) {
                            element.scrollIntoView({ behavior: 'smooth' });
                        }
                    }, 0);
                }
            })
            .catch(err => {
                console.error(`Error fetching doc content for ${filePath}:`, err);
                setDocContent({ html: `<p>Error loading content for ${filePath}.</p>`, markdown: '' });
                setError(`Failed to load content for ${filePath}.`);
                setIsLoading(false);
            });
    }, []);

    if (isLoading && !docContent.html) { // Show general loading only if nothing is displayed yet
        return <p>Loading documentation...</p>;
    }

    if (error && !references.length) { // Show error if initial load failed
        return <p style={{ color: 'red' }}>{error}</p>;
    }

    return (
        <div style={{ display: 'flex', height: 'calc(100vh - 200px)' }}>
            <nav style={{ width: '30%', borderRight: '1px solid #ccc', padding: '10px', overflowY: 'auto' }}>
                <h2>Documentation Index</h2>
                {references.length === 0 && !isLoading && <p>No references found.</p>}
                <ul>
                    {references.map(ref => (
                        <li key={ref.code} style={{ marginBottom: '10px' }}>
                            <strong>{ref.code}</strong>: {ref.description}
                            <button onClick={() => fetchDocContent(ref.link)} style={{ marginLeft: '10px' }}>View</button>
                        </li>
                    ))}
                </ul>
            </nav>
            <article style={{ width: '70%', padding: '20px', overflowY: 'auto' }}>
                {isLoading && <p>Loading content...</p>}
                {docContent.html ? (
                    <div dangerouslySetInnerHTML={{ __html: docContent.html }} />
                ) : (
                    !isLoading && <p>Select a document from the index to view its content.</p>
                )}
            </article>
        </div>
    );
}

export default DocExplorer;
