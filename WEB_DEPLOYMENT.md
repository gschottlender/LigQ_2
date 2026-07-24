# LigQ 2 public web deployment

LigQ 2 has two deployment modes in the same codebase:

- `local` is the default and preserves the complete application, first-run
  installer, custom databases, additional representations, BSI and unrestricted
  local search sizes.
- `web` is the restricted anonymous public interface. It is enabled only with
  `LIGQ_DEPLOYMENT_MODE=web`.

The web mode does not change the scientific behavior or options of the local
mode.

## Public search policy

The backend enforces the policy independently of the browser:

- ZINC is the only predicted-ligand provider.
- Available predicted searches are Morgan ECFP/Tanimoto with cached coverage
  from `0.4`, and Morgan Feature FCFP/Tanimoto with cached coverage from `0.5`.
  Users may raise, but not lower, those thresholds.
- **Known ligands only** skips the ZINC prediction and returns the PDB/ChEMBL
  evidence. The full web data package must still be ready before any search.
- Sequence, Nearest K (`1`–`15`) and Domain searches are available, with at
  least one method required.
- Each FASTA is limited to 100 records, 5 MiB and 500,000 total residues.
  Query identifiers must be unique.
- The server accepts one active search globally and does not queue a second
  search. An IP can have at most 20 accepted submissions per hour. A job is
  stopped after 60 minutes.
- Anonymous users can inspect and cancel only jobs belonging to their browser
  session. Completed results are retained for two hours. Uploads and temporary
  files are deleted at terminal status; failed and cancelled result artifacts
  are deleted immediately.
- Resource setup, custom databases, new representations and BSI are disabled.
- Runtime databases and predicted caches are mounted read-only.

## Test it locally without changing the local installation

Build the current source:

```bash
./docker/ligq-web.sh build
```

If `./databases` already contains the core data plus both ECFP and FCFP caches,
validate and reuse it read-only:

```bash
./docker/ligq-web.sh start-local-data
```

Open <http://127.0.0.1:18081>. The public stack has a separate Compose project,
job database, uploads, temporary files and result volumes. Only the existing
`./databases` directory is shared, and it is mounted read-only. The normal local
application on port 8080 is unaffected.

Stop the test stack with:

```bash
./docker/ligq-web.sh stop
```

## Prepare the production data volume

The production-style stack uses a dedicated named volume. Download the
mandatory core package and both predicted caches, then validate their manifests,
database fingerprints, thresholds and protein coverage:

```bash
./docker/ligq-web.sh prepare
```

This can take several hours or days and requires substantial disk space. The
operation is resumable because existing compatible files are reused. The data
volume is named `ligq2_web_databases` by default.

Start the application:

```bash
./docker/ligq-web.sh start
./docker/ligq-web.sh status
```

The default bind address is intentionally `127.0.0.1:18081`. Do not expose the
service directly to the Internet. Put it behind the faculty HTTPS reverse proxy
and set production values before publishing, for example:

```dotenv
LIGQ_WEB_BIND=127.0.0.1
LIGQ_WEB_PORT=18081
LIGQ_SESSION_SECRET=a-long-random-secret
LIGQ_SESSION_COOKIE_SECURE=true
```

The bundled Nginx overwrites client-supplied forwarding headers. If another
trusted reverse proxy sits in front of it, configure that proxy/Nginx trust
boundary so the API receives the real client address; otherwise per-IP rate
limits will see the proxy address.

Never run `docker compose -f compose.web.yml down -v` unless deletion of all
public application volumes is intentional.

## Immutable data validation

The same fail-closed validation can be run directly:

```bash
python validate_web_data.py --data-dir databases
```

The public frontend remains on a maintenance screen if the core files, either
representation, either complete five-file cache, cache fingerprint, cached
threshold coverage or protein coverage is missing or incompatible.

## Published images

The GitHub workflow publishes `main`, release and immutable `sha-<commit>`
container tags. For production, pin both API and frontend to the same immutable
SHA:

```dotenv
LIGQ_API_IMAGE=ghcr.io/gschottlender/ligq-2-api:sha-0123456
LIGQ_WEB_IMAGE=ghcr.io/gschottlender/ligq-2-web:sha-0123456
```
