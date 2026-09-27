# Deploying the aqdas MCP server

A runbook for the one hosted instance. The server is stateless and read-only —
no database, no volume, nothing to back up — so deploying it is: build an
image, ship it to the box, run it, and point the proxy at it.

## Where it runs

| | |
|---|---|
| Box | the Munnin VPS — `198.44.26.137` (`ssh munnin-vps`, user `agent`) |
| Container | `aqdas-<short-sha>`, on the Docker network `kamal` |
| Proxy | the box's existing `kamal-proxy` (container `kamal-proxy`), which also serves `munnin.lok.quest` |
| URL | `https://aqdas.lok.quest/mcp` — TLS issued by `kamal-proxy` |
| Auth | **none** — it is a public, read-only demo. See *Known debts*. |

The service is **not** managed by Kamal. It only borrows Kamal's proxy; it has
no `config/deploy.yml`, no registry, and no CI.

## Build

From the repo root, on any machine with Docker:

```sh
docker build -t aqdas:$(git rev-parse --short HEAD) .
```

The image derives its own corpus during the build (the `data/` tree is
gitignored): `fetch` pulls the Reference Library fragments, `parse` turns them
into citable records, and one `HybridRetriever` instantiation warms the
embedding matrix. So the build needs network access to `bahai.org` and to the
model host, and the runtime container needs neither.

## Ship

No registry: send the image straight to the daemon.

```sh
docker save aqdas:<sha> | ssh munnin-vps 'docker load'
```

## Run

```sh
ssh munnin-vps 'bash -s' < deploy/redeploy.sh aqdas:<sha>
```

`redeploy.sh` starts `aqdas-<sha>`, waits for its own `/health`, hands
`kamal-proxy` the new target, then removes the previous container. The proxy
health-checks `/health` before routing, so traffic only moves once the new
container answers.

## DNS (once)

`aqdas.lok.quest` must have an **A record to `198.44.26.137`**, with Cloudflare's
orange-cloud proxy **off** — otherwise the ACME challenge is intercepted before
it reaches the box and `--tls` never issues. The `--tls` step in `redeploy.sh`
fails until this record resolves.

## Verify

```sh
curl -sS -o /dev/null -w '%{http_code}\n' https://aqdas.lok.quest/health   # 200
```

Then a real round trip — `initialize` plus one `search` — from an MCP client,
not just the health endpoint.

## Rollback

Re-run `redeploy.sh` with the previous tag; the image is still in the daemon
until something prunes it. There is no one-command rollback (that is what
Kamal would have bought).

## Known debts

- **No auth.** Any caller can spend CPU — each query runs an ONNX embedding on
  a 2-vCPU box that also serves Munnin. Accepted for a public demo.
- **No swap on the box.** A memory spike has no cushion; a `/swapfile` was
  recommended for this host and never applied.
- **No CI.** Images are built and shipped by hand; nothing gates a push on the
  tests passing.
