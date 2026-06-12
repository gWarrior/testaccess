import { useCallback, useEffect, useRef, useState } from 'react';
import {
  clearCredentials,
  deleteFile,
  formatSize,
  hasCredentials,
  listFiles,
  listPlatforms,
  saveCredentials,
  uploadFiles,
} from './api.js';

function Login({ onLogin }) {
  const [user, setUser] = useState('');
  const [pass, setPass] = useState('');
  const [error, setError] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    saveCredentials(user, pass);
    try {
      await listPlatforms(); // validate credentials
      onLogin();
    } catch (err) {
      clearCredentials();
      setError(err.status === 401 ? 'Invalid credentials' : err.message);
    }
  };

  return (
    <div className="card login">
      <h1>Addressables Manager</h1>
      <form onSubmit={submit}>
        <label>
          User
          <input value={user} onChange={(e) => setUser(e.target.value)} autoFocus />
        </label>
        <label>
          Password
          <input
            type="password"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
          />
        </label>
        {error && <p className="error">{error}</p>}
        <button type="submit">Sign in</button>
      </form>
    </div>
  );
}

function UploadBox({ platform, onUploaded, onAuthError }) {
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const doUpload = async (files) => {
    if (!files || files.length === 0) return;
    setError('');
    setProgress(0);
    try {
      await uploadFiles(platform, files, setProgress);
      setProgress(null);
      onUploaded();
    } catch (err) {
      setProgress(null);
      if (err.status === 401) onAuthError();
      else setError(err.message);
    }
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="upload">
      <input
        ref={inputRef}
        type="file"
        multiple
        onChange={(e) => doUpload(e.target.files)}
      />
      {progress !== null && (
        <div className="progress">
          <div className="bar" style={{ width: `${progress}%` }} />
          <span>{progress}%</span>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(hasCredentials());
  const [platforms, setPlatforms] = useState([]);
  const [selected, setSelected] = useState(null);
  const [files, setFiles] = useState([]);
  const [error, setError] = useState('');
  const [newPlatform, setNewPlatform] = useState('');

  const onAuthError = useCallback(() => {
    clearCredentials();
    setAuthed(false);
  }, []);

  const refreshPlatforms = useCallback(async () => {
    try {
      const { platforms } = await listPlatforms();
      setPlatforms(platforms);
    } catch (err) {
      if (err.status === 401) onAuthError();
      else setError(err.message);
    }
  }, [onAuthError]);

  const refreshFiles = useCallback(
    async (platform) => {
      if (!platform) return;
      try {
        const { files } = await listFiles(platform);
        setFiles(files);
      } catch (err) {
        if (err.status === 401) onAuthError();
        else setError(err.message);
      }
    },
    [onAuthError],
  );

  useEffect(() => {
    if (authed) refreshPlatforms();
  }, [authed, refreshPlatforms]);

  useEffect(() => {
    if (selected) refreshFiles(selected);
  }, [selected, refreshFiles]);

  const onDelete = async (filename) => {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
      await deleteFile(selected, filename);
      refreshFiles(selected);
      refreshPlatforms();
    } catch (err) {
      if (err.status === 401) onAuthError();
      else setError(err.message);
    }
  };

  const onUploaded = () => {
    refreshFiles(selected);
    refreshPlatforms();
  };

  const createPlatform = (e) => {
    e.preventDefault();
    const name = newPlatform.trim();
    if (!name) return;
    setSelected(name);
    setNewPlatform('');
  };

  if (!authed) return <Login onLogin={() => setAuthed(true)} />;

  return (
    <div className="app">
      <header>
        <h1>Addressables Manager</h1>
        <button className="ghost" onClick={onAuthError}>
          Sign out
        </button>
      </header>

      {error && <p className="error global">{error}</p>}

      <div className="layout">
        <aside className="card">
          <h2>Platforms</h2>
          <ul className="platforms">
            {platforms.map((p) => (
              <li
                key={p.name}
                className={p.name === selected ? 'active' : ''}
                onClick={() => setSelected(p.name)}
              >
                <span>{p.name}</span>
                <small>
                  {p.fileCount} files · {formatSize(p.totalSize)}
                </small>
              </li>
            ))}
            {platforms.length === 0 && <li className="empty">No platforms yet</li>}
          </ul>
          <form className="new-platform" onSubmit={createPlatform}>
            <input
              placeholder="New platform (e.g. Android)"
              value={newPlatform}
              onChange={(e) => setNewPlatform(e.target.value)}
            />
            <button type="submit">+</button>
          </form>
        </aside>

        <main className="card">
          {selected ? (
            <>
              <h2>{selected}</h2>
              <UploadBox
                platform={selected}
                onUploaded={onUploaded}
                onAuthError={onAuthError}
              />
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Size</th>
                    <th>Modified</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((f) => (
                    <tr key={f.name}>
                      <td>
                        <a href={f.url} target="_blank" rel="noreferrer">
                          {f.name}
                        </a>
                      </td>
                      <td>{formatSize(f.size)}</td>
                      <td>{new Date(f.modified).toLocaleString()}</td>
                      <td>
                        <button className="danger" onClick={() => onDelete(f.name)}>
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                  {files.length === 0 && (
                    <tr>
                      <td colSpan={4} className="empty">
                        No files. Upload bundles, catalog and hash here.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </>
          ) : (
            <p className="placeholder">Select or create a platform to manage files.</p>
          )}
        </main>
      </div>
    </div>
  );
}
