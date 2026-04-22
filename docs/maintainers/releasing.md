# Releasing `@zerople/hermes-shim-http`

Maintainer notes for publishing the npm package.

## What gets published

The npm package publishes only the runtime pieces needed by end users:

- `bin/hermes-shim-http.js`
- `hermes_shim_http/`
- `requirements.txt`
- `README.md`
- `CHANGELOG.md`
- `LICENSE`

On first run, the Node launcher bootstraps a cached Python virtual environment and installs the pinned Python dependencies.

## Repository setup

Before the first public release, configure:

1. npm package ownership for the `@zerople` scope
2. npm trusted publishing for this GitHub repository
   - add the GitHub repo/workflow as a trusted publisher in npm
   - publish workflow expects OIDC (`id-token: write`) + `npm publish --provenance`
3. recommended repository settings:
   - protect `main`
   - restrict who can create release tags if needed

## Local preflight checklist

Run all of these before pushing a release:

```bash
python3 -m pytest -q
node --test tests/*.test.js
node bin/hermes-shim-http.js --help
npm pack
npm publish --dry-run --access public
```

## Version bump checklist

Update these together for every release:

- `package.json`
- `pyproject.toml`
- `hermes_shim_http/__init__.py`
- `CHANGELOG.md`
- tests that assert the version string
- optional per-release notes (if maintained), e.g. `docs/releases/vX.Y.Z.md`

The publish workflow also validates that these versions match, and if the workflow was triggered by a tag push, it verifies the tag version matches the package version.

## Recommended release flow

For the very first repository upload, use a two-step release flow:

### Step 1: publish the repository contents only

Push `main` first and let CI run without triggering npm publish yet:

```bash
git push -u origin main
```

This keeps the first upload clean and gives you one chance to inspect the GitHub repo, Actions, and rendered README before any package release happens.

### Step 2: publish the npm release only after CI is green

After the repository is live and CI looks good, create the release tag:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

That tag push triggers `.github/workflows/publish-npm.yml`.

The workflow is configured for npm trusted publishing via GitHub OIDC (`id-token: write` + `npm publish --provenance`). If trusted publishing is not configured yet on npm, the release job will fail until you add the trusted publisher or temporarily switch back to token-based publishing.

### Why this is the recommended default

- first push creates the repo history cleanly
- CI validates the default branch before release
- npm publish is isolated to the tag push
- rollback decisions are simpler if something looks off on GitHub

If GitHub Actions is unavailable, manual fallback is:

```bash
npm publish --access public
```

If your npm account enforces 2FA for publish, local/manual publish also needs either an interactive OTP flow or an allowed automation/granular token with the right publish permissions.

## Workflow summary

- CI: `.github/workflows/ci.yml`
- Publish: `.github/workflows/publish-npm.yml`

Current publish behavior:

- package name: `@zerople/hermes-shim-http`
- package access: `public`
- publish method: npm trusted publishing (recommended)
- publish triggers:
  - tag push matching `v*`
  - manual workflow dispatch
