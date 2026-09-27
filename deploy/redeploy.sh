#!/usr/bin/env bash
# Deploy or redeploy the aqdas MCP server on the Munnin box.
#
# Runs ON THE BOX. The image must already be in the local Docker daemon.
#
#   deploy/redeploy.sh <image-ref>
#
# Blue-green by container name, so the proxy never points at a half-started
# container: start the new one, wait for its own /health, hand the proxy to it,
# and only then remove the previous one. `kamal-proxy` is driven directly —
# this service is not managed by Kamal, it just borrows the proxy.
set -euo pipefail

IMAGE="${1:?usage: redeploy.sh <image-ref>}"
PORT="${AQDAS_PORT:-8300}"
NETWORK="${AQDAS_NETWORK:-kamal}"
DOMAIN="${AQDAS_DOMAIN:-aqdas.lok.quest}"
SERVICE=aqdas
SHA="${IMAGE##*:}"
NEW="aqdas-$SHA"

echo "==> start $NEW from $IMAGE"
docker rm -f "$NEW" >/dev/null 2>&1 || true
docker run -d --name "$NEW" --network "$NETWORK" --restart unless-stopped "$IMAGE" >/dev/null

echo "==> wait for /health inside the container"
healthy=""
for i in $(seq 1 30); do
  if docker exec "$NEW" python -c "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:$PORT/health', timeout=2).status == 200 else 1)" 2>/dev/null; then
    healthy="yes"; echo "    healthy after ${i}s"; break
  fi
  sleep 1
done
if [ -z "$healthy" ]; then
  echo "    never became healthy — logs:"; docker logs --tail 50 "$NEW"; exit 1
fi

echo "==> hand the proxy to $NEW"
docker exec kamal-proxy kamal-proxy deploy "$SERVICE" \
  --target "$NEW:$PORT" \
  --host "$DOMAIN" \
  --tls \
  --health-check-path /health

echo "==> remove every earlier aqdas container"
for c in $(docker ps -a --format '{{.Names}}' | grep -E '^aqdas-' | grep -vx "$NEW" || true); do
  echo "    removing $c"; docker rm -f "$c" >/dev/null
done

echo "==> done — https://$DOMAIN/mcp"
docker exec kamal-proxy kamal-proxy list
