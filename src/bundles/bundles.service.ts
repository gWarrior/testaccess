import { Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as fs from 'node:fs';
import * as path from 'node:path';

export interface StoredFile {
  name: string;
  size: number;
  modified: string;
  url: string;
}

@Injectable()
export class BundlesService {
  private readonly storageDir: string;

  constructor(private readonly config: ConfigService) {
    this.storageDir = this.config.get<string>('storageDir')!;
    fs.mkdirSync(this.storageDir, { recursive: true });
  }

  // Restrict path segments to a safe charset so a value can't escape the
  // storage root (path traversal) or use reserved dot names.
  sanitizeSegment(value: string): string {
    return String(value || '')
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .replace(/^\.+/, '_');
  }

  // Resolve (and create) the directory for a given platform/profile.
  platformDir(platform: string): { safe: string; dir: string } {
    const safe = this.sanitizeSegment(platform) || 'default';
    const dir = path.join(this.storageDir, safe);
    fs.mkdirSync(dir, { recursive: true });
    return { safe, dir };
  }

  private toUrl(platform: string, name: string): string {
    return `/content/${platform}/${encodeURIComponent(name)}`;
  }

  // List all platforms/profiles (top-level directories) with summary info.
  listPlatforms(): {
    platforms: { name: string; fileCount: number; totalSize: number }[];
  } {
    const platforms = fs
      .readdirSync(this.storageDir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => {
        const dir = path.join(this.storageDir, e.name);
        const files = fs
          .readdirSync(dir, { withFileTypes: true })
          .filter((f) => f.isFile());
        const totalSize = files.reduce(
          (sum, f) => sum + fs.statSync(path.join(dir, f.name)).size,
          0,
        );
        return { name: e.name, fileCount: files.length, totalSize };
      });
    return { platforms };
  }

  list(platform: string): { platform: string; files: StoredFile[] } {
    const { safe, dir } = this.platformDir(platform);
    const files = fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => {
        const stat = fs.statSync(path.join(dir, e.name));
        return {
          name: e.name,
          size: stat.size,
          modified: stat.mtime.toISOString(),
          url: this.toUrl(safe, e.name),
        };
      });
    return { platform: safe, files };
  }

  describeUploaded(platform: string, files: Express.Multer.File[]): StoredFile[] {
    const { safe } = this.platformDir(platform);
    return files.map((f) => ({
      name: f.filename,
      size: f.size,
      modified: new Date().toISOString(),
      url: this.toUrl(safe, f.filename),
    }));
  }

  remove(platform: string, filename: string): string {
    const { dir } = this.platformDir(platform);
    const name = this.sanitizeSegment(filename);
    const target = path.join(dir, name);
    if (!fs.existsSync(target)) {
      throw new NotFoundException('File not found');
    }
    fs.unlinkSync(target);
    return name;
  }
}
