import * as schema from './schema';

export function getDb() {
  throw new Error(
    'No database is configured for the Next.js frontend. Add a Next-compatible database adapter before using getDb().',
  );
}

export { schema };
