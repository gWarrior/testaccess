// Thin API client for the Addressables server. Credentials are kept in
// sessionStorage as a base64 Basic-auth token and sent on every request.

const TOKEN_KEY = 'addr-admin-token';

export function saveCredentials(user, pass) {
  const token = btoa(`${user}:${pass}`);
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearCredentials() {
  sessionStorage.removeItem(TOKEN_KEY);
}

export function hasCredentials() {
  return Boolean(sessionStorage.getItem(TOKEN_KEY));
}

function authHeader() {
  const token = sessionStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Basic ${token}` } : {};
}

async function handle(res) {
  if (res.status === 401) {
    const err = new Error('Unauthorized');
    err.status = 401;
    throw err;
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.message) message = body.message;
      else if (body?.error) message = body.error;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return res.json();
}

export async function listPlatforms() {
  return handle(await fetch('/api/bundles', { headers: authHeader() }));
}

export async function listFiles(platform) {
  return handle(
    await fetch(`/api/bundles/${encodeURIComponent(platform)}`, {
      headers: authHeader(),
    }),
  );
}

export async function uploadFiles(platform, files, onProgress) {
  // XHR is used (instead of fetch) so we can report upload progress.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const f of files) form.append('files', f);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/bundles/${encodeURIComponent(platform)}`);
    const token = sessionStorage.getItem(TOKEN_KEY);
    if (token) xhr.setRequestHeader('Authorization', `Basic ${token}`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else if (xhr.status === 401) {
        const err = new Error('Unauthorized');
        err.status = 401;
        reject(err);
      } else {
        reject(new Error(`Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(form);
  });
}

export async function deleteFile(platform, filename) {
  return handle(
    await fetch(
      `/api/bundles/${encodeURIComponent(platform)}/${encodeURIComponent(filename)}`,
      { method: 'DELETE', headers: authHeader() },
    ),
  );
}

export function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}
