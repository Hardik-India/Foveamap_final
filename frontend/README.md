# FoveaMap frontend

This frontend is a standard Next.js, React and TypeScript application.

## Windows PowerShell

```powershell
pnpm install --frozen-lockfile
pnpm dev
```

Open http://localhost:5173.

Useful checks:

```powershell
pnpm test
pnpm exec tsc --noEmit
pnpm build
```

The dashboard supports procedural scenes, CSV/KITTI scan imports, neural demo inference, adaptive-grid inspection and JSON/CSV exports.
