import * as crypto from 'node:crypto';

// Timing-safe string comparison to avoid leaking credential length/content
// through response timing.
export function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) {
    crypto.timingSafeEqual(bufA, bufA);
    return false;
  }
  return crypto.timingSafeEqual(bufA, bufB);
}

// Parse and validate an HTTP "Authorization: Basic ..." header against the
// expected credentials. Returns true when the credentials match.
export function checkBasicAuth(
  authorization: string | undefined,
  expectedUser: string,
  expectedPass: string,
): boolean {
  if (!authorization) return false;
  const [scheme, encoded] = authorization.split(' ');
  if (scheme !== 'Basic' || !encoded) return false;

  let user = '';
  let pass = '';
  try {
    const decoded = Buffer.from(encoded, 'base64').toString('utf8');
    const idx = decoded.indexOf(':');
    if (idx === -1) return false;
    user = decoded.slice(0, idx);
    pass = decoded.slice(idx + 1);
  } catch {
    return false;
  }

  const userOk = safeEqual(user, expectedUser);
  const passOk = safeEqual(pass, expectedPass);
  return userOk && passOk;
}
