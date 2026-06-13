# Unity Addressables Server

A NestJS server that lets a Unity app **upload Addressables bundles** (bundles,
catalog, hash) and **serve them back** for remote loading. All API routes are
protected with **HTTP Basic auth**, and a built‑in **React admin page** lets you
browse and manage uploaded content.

## Features

- 📤 Upload bundles/catalog/hash per platform (`Android`, `iOS`, `StandaloneWindows64`, …)
- 📥 Serve content back to Unity over `/content/...` (the Addressables RemoteLoadPath)
- 🔐 HTTP Basic auth on all `/api` routes (downloads optionally protected too)
- 🖥️ React management UI for browsing, uploading and deleting bundles
- 📚 Swagger / OpenAPI docs at `/api/docs`
- 🎮 Unity upload example (Editor script) + auth example for downloads

## Project layout

```
src/                 NestJS server (TypeScript)
  auth/              HTTP Basic auth guard + helpers
  bundles/           Upload / list / delete controller + service
  config/            Env-based configuration
  main.ts            Bootstrap, Swagger, static serving
client/              React (Vite) admin UI
examples/unity/      Unity upload + auth examples (C#)
```

## Quick start

```bash
# 1. Configure credentials
cp .env.example .env        # then edit API_USER / API_PASS

# 2. Install + build the admin UI (served by the server at /)
cd client && npm install && npm run build && cd ..

# 3. Install + run the server
npm install
npm run build
npm run start:prod          # or: npm run start:dev  (watch mode)
```

Then open:

- Admin UI: <http://localhost:3001/>
- Swagger:  <http://localhost:3001/api/docs>

## Run with Docker

Frontend and backend each have their own image; `docker-compose` wires them
together. The frontend (nginx) serves the React UI and reverse‑proxies
`/api`, `/content` and `/health` to the backend, so everything is reachable on
a single port.

```bash
cp .env.example .env        # set API_USER / API_PASS (used by compose)
docker compose up --build
```

Then open:

- App + API: <http://localhost:8080/>
- Swagger:   <http://localhost:8080/api/docs>

Uploaded bundles are persisted in the `bundles` named volume. Point Unity at
`http://YOUR_HOST:8080/...` for both upload and `RemoteLoadPath`.

Compose env vars (all optional, see `.env.example`):

| Variable            | Default    | Description                           |
| ------------------- | ---------- | ------------------------------------- |
| `WEB_PORT`          | `8080`     | Host port for the frontend            |
| `API_USER`/`API_PASS` | `unity`/`changeme` | Basic-auth credentials      |
| `PROTECT_DOWNLOADS` | `false`    | Require auth on `/content` downloads  |
| `MAX_FILE_SIZE`     | `536870912`| Max upload size per file (bytes)      |

> nginx is configured with `client_max_body_size 1024m`; raise it in
> `client/nginx.conf` if your bundles are larger.

## Configuration

All settings come from environment variables (see `.env.example`):

| Variable            | Default        | Description                                     |
| ------------------- | -------------- | ----------------------------------------------- |
| `PORT`              | `3001`         | HTTP port                                       |
| `API_USER`          | `unity`        | Basic-auth username                             |
| `API_PASS`          | `changeme`     | Basic-auth password                             |
| `API_REALM`         | `Addressables` | Basic-auth realm                                |
| `STORAGE_DIR`       | `./storage`    | Where uploaded files are stored                 |
| `MAX_FILE_SIZE`     | `536870912`    | Max upload size per file (bytes, default 512MB) |
| `PROTECT_DOWNLOADS` | `false`        | Require basic auth on `/content` downloads too  |

## API

All `/api` routes require an `Authorization: Basic <base64(user:pass)>` header.

| Method   | Path                              | Description                          |
| -------- | --------------------------------- | ------------------------------------ |
| `POST`   | `/api/bundles/:platform`          | Upload files (multipart, field `files`) |
| `GET`    | `/api/bundles`                    | List all platforms                   |
| `GET`    | `/api/bundles/:platform`          | List files for a platform            |
| `DELETE` | `/api/bundles/:platform/:filename`| Delete a file                        |
| `GET`    | `/content/:platform/:filename`    | Download a file (Unity RemoteLoadPath) |
| `GET`    | `/health`                         | Health check                         |

### Example with curl

```bash
# Upload
curl -u unity:changeme \
  -F "files=@catalog.json" \
  -F "files=@assets_all_xxxx.bundle" \
  http://localhost:3001/api/bundles/Android

# List
curl -u unity:changeme http://localhost:3001/api/bundles/Android

# Download (what Unity does)
curl http://localhost:3001/content/Android/catalog.json
```

## Unity integration

### 1. Point Addressables at the server

In **Addressables Profiles** set your remote paths:

- **RemoteBuildPath**: `ServerData/[BuildTarget]` (default)
- **RemoteLoadPath**: `http://YOUR_HOST:3001/content/[BuildTarget]`

Build content: *Window → Asset Management → Addressables → Groups → Build → New Build → Default Build Script*.

### 2. Upload the build

Copy [`examples/unity/AddressablesUploader.cs`](examples/unity/AddressablesUploader.cs)
into an `Assets/Editor/` folder, set `ServerUrl` / `ApiUser` / `ApiPass`, then run
**Tools → Addressables → Upload To Server**. It uploads everything in
`ServerData/<BuildTarget>/` to `/api/bundles/<BuildTarget>` with basic auth.

### 3. (Optional) Protected downloads

If you start the server with `PROTECT_DOWNLOADS=true`, downloads also need
credentials. Add [`examples/unity/AddressablesAuth.cs`](examples/unity/AddressablesAuth.cs)
to your project — it registers an `Addressables.WebRequestOverride` that attaches
the `Authorization` header to every download.

## Development

```bash
# Run the API with hot reload
npm run start:dev

# Run the admin UI with hot reload (proxies /api to localhost:3001)
cd client && npm run dev      # http://localhost:5173
```

> **Security note:** Basic auth only protects credentials in transit when used
> over **HTTPS**. Put this server behind TLS (a reverse proxy such as Nginx or
> Caddy) before exposing it publicly, and change the default credentials.
