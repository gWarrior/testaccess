import * as path from 'node:path';

// Application configuration loaded from environment variables.
// Defaults are suitable for local development only — set real credentials
// via the environment in production (see .env.example).
export interface AppConfig {
  port: number;
  auth: {
    user: string;
    pass: string;
    realm: string;
  };
  storageDir: string;
  maxFileSize: number;
  protectDownloads: boolean;
}

export default (): AppConfig => ({
  port: Number(process.env.PORT) || 3001,
  auth: {
    user: process.env.API_USER || 'unity',
    pass: process.env.API_PASS || 'changeme',
    realm: process.env.API_REALM || 'Addressables',
  },
  storageDir: process.env.STORAGE_DIR
    ? path.resolve(process.env.STORAGE_DIR)
    : path.resolve(process.cwd(), 'storage'),
  // Max upload size per file (bytes). Default 512 MB.
  maxFileSize: Number(process.env.MAX_FILE_SIZE) || 512 * 1024 * 1024,
  // When true, downloading content also requires basic auth.
  protectDownloads: process.env.PROTECT_DOWNLOADS === 'true',
});
