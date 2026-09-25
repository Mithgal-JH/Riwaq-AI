# Riwaq-AI

AI services and recommendation systems for Riwaq.

## Railway Deployment

The Riwaq-AI Railway service uses the repository-root `Dockerfile` and runs three
processes in one container:

- `ai-gateway` is the only publicly exposed process and listens on Railway's
  `$PORT` (currently `8080`).
- `content-analysis` listens only on `127.0.0.1:8001`.
- `post-recommendation` listens only on `127.0.0.1:8002`.

The gateway keeps its existing defaults for the two internal services:

```text
CONTENT_ANALYSIS_URL=http://127.0.0.1:8001
POST_RECOMMENDATION_URL=http://127.0.0.1:8002
```

Person recommendations remain an external service. Set
`PERSON_RECOMMENDATION_URL` to the existing person-recommendation Railway
private URL; do not point it at localhost.

### Required Railway Settings

For the existing Riwaq-AI Railway service:

1. Set **Root Directory** to the repository root (`/`).
2. Set **Dockerfile Path** to `/Dockerfile` (or leave it for Railway's root
   Dockerfile autodetection).
3. Do not set a custom start command; the Dockerfile entrypoint starts the
   processes.
4. Leave Railway's `PORT` variable managed by Railway.
5. Configure `PERSON_RECOMMENDATION_URL` with the existing private Railway
   URL.
6. `CONTENT_ANALYSIS_URL` and `POST_RECOMMENDATION_URL` may be omitted. If
   configured, they must remain `http://127.0.0.1:8001` and
   `http://127.0.0.1:8002` respectively.

Ports `8001` and `8002` are intentionally not exposed by the Dockerfile.

The startup script waits for both internal health endpoints before starting
the gateway. If an internal process or the gateway exits, the remaining
processes are terminated and the container exits non-zero.

## Local Development

`compose.yaml` continues to provide the existing multi-container local setup,
including the person-recommendation container and Docker service-name URLs.
