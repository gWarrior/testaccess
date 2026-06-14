import type { MulterOptions } from '@nestjs/platform-express/multer/interfaces/multer-options.interface';
import { diskStorage } from 'multer';
import type { Request } from 'express';
import * as fs from 'node:fs';
import * as path from 'node:path';
import configuration from '../config/configuration';

// FilesInterceptor options are evaluated when the controller class is defined,
// before DI is available, so we read configuration directly here.
const cfg = configuration();

function sanitizeSegment(value: string): string {
  return String(value || '')
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .replace(/^\.+/, '_');
}

export const multerOptions: MulterOptions = {
  storage: diskStorage({
    destination(req: Request, _file, cb) {
      try {
        const platform =
          sanitizeSegment(String(req.params.platform)) || 'default';
        const dir = path.join(cfg.storageDir, platform);
        fs.mkdirSync(dir, { recursive: true });
        cb(null, dir);
      } catch (err) {
        cb(err as Error, '');
      }
    },
    filename(_req, file, cb) {
      cb(null, sanitizeSegment(file.originalname));
    },
  }),
  limits: { fileSize: cfg.maxFileSize },
};
