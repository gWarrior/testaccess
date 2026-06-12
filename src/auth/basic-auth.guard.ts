import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { Request, Response } from 'express';
import { checkBasicAuth } from './basic-auth.util';

// Guard that enforces HTTP Basic authentication on the routes it protects.
@Injectable()
export class BasicAuthGuard implements CanActivate {
  constructor(private readonly config: ConfigService) {}

  canActivate(context: ExecutionContext): boolean {
    const req = context.switchToHttp().getRequest<Request>();
    const res = context.switchToHttp().getResponse<Response>();

    const user = this.config.get<string>('auth.user')!;
    const pass = this.config.get<string>('auth.pass')!;
    const realm = this.config.get<string>('auth.realm')!;

    if (checkBasicAuth(req.headers.authorization, user, pass)) {
      return true;
    }

    res.setHeader(
      'WWW-Authenticate',
      `Basic realm="${realm}", charset="UTF-8"`,
    );
    throw new UnauthorizedException('Unauthorized');
  }
}
