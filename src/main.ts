import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import { NestExpressApplication } from '@nestjs/platform-express';
import { ConfigService } from '@nestjs/config';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import * as fs from 'node:fs';
import * as path from 'node:path';
import type { Request, Response, NextFunction } from 'express';
import { AppModule } from './app.module';
import { checkBasicAuth } from './auth/basic-auth.util';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  const config = app.get(ConfigService);

  const port = config.get<number>('port')!;
  const storageDir = config.get<string>('storageDir')!;
  const protectDownloads = config.get<boolean>('protectDownloads')!;
  const user = config.get<string>('auth.user')!;
  const pass = config.get<string>('auth.pass')!;
  const realm = config.get<string>('auth.realm')!;

  fs.mkdirSync(storageDir, { recursive: true });

  // Allow the React admin UI (served from the same origin) to call the API.
  app.enableCors();

  // Swagger / OpenAPI documentation at /api/docs.
  const swaggerConfig = new DocumentBuilder()
    .setTitle('Unity Addressables Server')
    .setDescription(
      'Upload and serve Unity Addressables bundles. All /api routes require HTTP Basic auth.',
    )
    .setVersion('1.0')
    .addBasicAuth()
    .build();
  const document = SwaggerModule.createDocument(app, swaggerConfig);
  SwaggerModule.setup('api/docs', app, document);

  // Optionally require basic auth on the download path too. Unity Addressables
  // can supply credentials via a WebRequestOverride.
  if (protectDownloads) {
    app.use('/content', (req: Request, res: Response, next: NextFunction) => {
      if (checkBasicAuth(req.headers.authorization, user, pass)) {
        return next();
      }
      res.setHeader(
        'WWW-Authenticate',
        `Basic realm="${realm}", charset="UTF-8"`,
      );
      res.status(401).json({ error: 'Unauthorized' });
    });
  }

  // Serve uploaded content back to Unity for remote loading.
  app.useStaticAssets(storageDir, { prefix: '/content/', index: false });

  // Serve the built React management UI at the root (if it has been built).
  const clientDist = path.resolve(process.cwd(), 'client', 'dist');
  if (fs.existsSync(clientDist)) {
    app.useStaticAssets(clientDist);
  }

  await app.listen(port);
  console.log(`Addressables server listening on http://localhost:${port}`);
  console.log(`Storage directory: ${storageDir}`);
  console.log(`Download auth: ${protectDownloads ? 'required' : 'public'}`);
}

bootstrap();
