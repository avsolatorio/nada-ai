# OpenSearch host setup (no Docker)

How to run **OpenSearch** and the **nada-ai API** directly on a Linux server, without Docker or Podman, as system
services. When both are running, continue with [creating the indexes](./opensearch-setup-guide.md#5-create-the-indexes)
in the main OpenSearch guide.

Is Docker allowed on the server? Then the [Docker setup](./opensearch-setup-guide.md#4-start-the-local-docker-stack)
is quicker. On a Mac, OpenSearch has no official native build (Homebrew's lacks the k-NN plugin), so run only
OpenSearch in Docker and the API on the host, as in
[Run the API on the host](./opensearch-setup-guide.md#run-the-api-on-the-host).

**Contents**

1. [Plan](#1-plan)
2. [Install OpenSearch](#2-install-opensearch)
3. [Secure OpenSearch and create the nada-ai user](#3-secure-opensearch-and-create-the-nada-ai-user)
4. [Install nada-ai](#4-install-nada-ai)
5. [Configure `.env`](#5-configure-env)
6. [Run the API as a service](#6-run-the-api-as-a-service)
7. [Reverse proxy](#7-reverse-proxy)
8. [Next steps](#8-next-steps)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Plan

| Item | Suggested value |
|---|---|
| OS | RHEL / Rocky / Alma 8 or 9, or Ubuntu 22.04 / 24.04, x86_64 |
| RAM | 8 GB minimum for both on one host (OpenSearch heap 2 to 4 GB, API with Harrier about 2 GB) |
| OpenSearch | 3.x (nada-ai is tuned for 3.6), listening on `127.0.0.1:9200` unless other hosts need it |
| API | uvicorn on `127.0.0.1:8020`, one worker, behind Apache or nginx |
| nada-ai checkout | `/opt/nada-ai` (any path works; adjust the commands below) |
| Caches | `/var/cache/uv`, `/var/cache/huggingface`, off a small `/` partition |
| Service user | `nada-ai`, owning the checkout, `.venv`, `config/`, `data/` and the caches |

---

## 2. Install OpenSearch

Follow the official instructions for your OS: [OpenSearch installation](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/index/).
Use the **full** distribution (it includes the k-NN, security and ML Commons plugins), not the `-min` one.
A summary of each route:

### RHEL family (RPM)

```bash
sudo curl -SL https://artifacts.opensearch.org/releases/bundle/opensearch/3.x/opensearch-3.x.repo \
  -o /etc/yum.repos.d/opensearch-3.x.repo
sudo dnf clean all
sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD='<strong-password>' dnf install -y opensearch
```

### Debian / Ubuntu (APT)

Add the OpenSearch APT repository as the official guide shows, then:

```bash
sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD='<strong-password>' apt-get install -y opensearch
```

### Tarball (any Linux, no package manager)

Download `opensearch-3.6.0-linux-x64.tar.gz` from the OpenSearch downloads page, unpack it under `/opt/opensearch`,
run it as a non-root user, and write your own systemd unit for `bin/opensearch`.

### Host settings

```bash
# Required by OpenSearch: persist across reboots
echo 'vm.max_map_count=262144' | sudo tee /etc/sysctl.d/99-opensearch.conf
sudo sysctl --system
```

Heap, in `/etc/opensearch/jvm.options.d/heap.options` (half the RAM you give OpenSearch; at most about 31 GB):

```text
-Xms2g
-Xmx2g
```

In `/etc/opensearch/opensearch.yml`, for a single node:

```yaml
cluster.name: nada-ai
node.name: node-1
network.host: 127.0.0.1        # or the private IP if the API runs on another host
http.port: 9200
discovery.type: single-node
path.data: /var/lib/opensearch
path.logs: /var/log/opensearch
```

Start it and check it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now opensearch
curl -sk -u admin:'<strong-password>' https://localhost:9200
curl -sk -u admin:'<strong-password>' https://localhost:9200/_cat/plugins | grep -E 'knn|security|ml'
```

The package install sets up the security plugin with **demo TLS certificates**. They are fine for a first test, not
for production: replace them with your own (see
[OpenSearch security configuration](https://docs.opensearch.org/latest/security/configuration/index/)).

---

## 3. Secure OpenSearch and create the nada-ai user

Give nada-ai its own user with only what it needs, rather than the `admin` account:

```bash
OS=https://localhost:9200
ADMIN="admin:<strong-password>"

curl -sk -u "$ADMIN" -X PUT "$OS/_plugins/_security/api/roles/nada_ai" \
  -H 'content-type: application/json' -d '{
    "cluster_permissions": ["cluster_monitor", "cluster_composite_ops", "cluster_manage_index_templates"],
    "index_permissions": [{"index_patterns": ["nada-metadata*"], "allowed_actions": ["indices_all"]}]
  }'

curl -sk -u "$ADMIN" -X PUT "$OS/_plugins/_security/api/internalusers/nada_ai" \
  -H 'content-type: application/json' -d '{"password": "<nada-ai-password>"}'

curl -sk -u "$ADMIN" -X PUT "$OS/_plugins/_security/api/rolesmapping/nada_ai" \
  -H 'content-type: application/json' -d '{"users": ["nada_ai"]}'

# Check: should return cluster info, not 401/403
curl -sk -u 'nada_ai:<nada-ai-password>' "$OS/_cluster/health?pretty"
```

If you change `NADA_INDEX_NAME`, change the index pattern to match. `cluster_composite_ops` covers bulk writes and
scrolls, which OpenSearch treats as cluster-level actions. For the
[`opensearch_ml` embedding backend](./opensearch-setup-guide.md#option-b-opensearch-ml-commons-opensearch_ml), also add
`cluster_manage_pipelines` and the ML Commons permissions for your model.

Copy the CA certificate that signed OpenSearch's HTTP certificate to where nada-ai can read it, e.g.
`/etc/nada-ai/opensearch-root-ca.pem` (with the demo certificates it is `/etc/opensearch/root-ca.pem`).

---

## 4. Install nada-ai

```bash
sudo useradd --system --home /opt/nada-ai --shell /sbin/nologin nada-ai
sudo mkdir -p /opt/nada-ai /var/cache/uv /var/cache/huggingface
sudo chown -R nada-ai: /opt/nada-ai /var/cache/uv /var/cache/huggingface

# uv (https://docs.astral.sh/uv/getting-started/installation/), installed system-wide
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh

sudo -u nada-ai git clone <nada-ai repository URL> /opt/nada-ai
cd /opt/nada-ai
sudo -u nada-ai env UV_CACHE_DIR=/var/cache/uv uv sync --extra local
```

On a CPU-only Linux server, `uv sync` also pulls the CUDA build of PyTorch and its NVIDIA libraries (several GB).
To skip them, install without those packages, then add `torch` on its own:

```bash
sudo -u nada-ai env UV_CACHE_DIR=/var/cache/uv uv sync --extra local \
  --no-install-package torch \
  --no-install-package nvidia-cudnn-cu13 \
  --no-install-package nvidia-cusparselt-cu13 \
  --no-install-package nvidia-nccl-cu13 \
  --no-install-package nvidia-nvshmem-cu13 \
  --no-install-package cuda-bindings \
  --no-install-package cuda-toolkit \
  --no-install-package triton
sudo -u nada-ai env UV_CACHE_DIR=/var/cache/uv uv pip install torch --no-deps
```

Use the same flags on every later `uv sync`; a plain `uv sync` puts the CUDA packages back. The package list follows
`uv.lock`: if a PyTorch upgrade changes the `nvidia-*` or `cuda-*` names, update it.

### Get the Harrier model

If the server can reach Hugging Face:

```bash
sudo -u nada-ai env HF_HOME=/var/cache/huggingface UV_CACHE_DIR=/var/cache/uv \
  uv run python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('microsoft/harrier-oss-v1-270m')"
```

If it cannot (the model weights come from a Hugging Face CDN that some firewalls block), download the model on a
machine that can, with nada-ai installed, and copy the folder over:

```bash
# on a machine with access
uv run hf download microsoft/harrier-oss-v1-270m --local-dir harrier-oss-v1-270m
tar czf harrier-oss-v1-270m.tgz harrier-oss-v1-270m

# on the server, after copying the archive over
sudo tar xzf harrier-oss-v1-270m.tgz -C /var/cache/huggingface
sudo chown -R nada-ai: /var/cache/huggingface/harrier-oss-v1-270m
ls -lh /var/cache/huggingface/harrier-oss-v1-270m/model.safetensors   # about 500 MB
```

Then set `NADA_EMBEDDING_MODEL_ID=/var/cache/huggingface/harrier-oss-v1-270m`.

---

## 5. Configure `.env`

```bash
cd /opt/nada-ai
sudo -u nada-ai cp .env.example .env
sudo -u nada-ai mkdir -p data/nada-discovery
sudo chmod 600 .env
```

The OpenSearch-specific values (everything else is as in
[the main guide](./opensearch-setup-guide.md#3-configure-env)):

```bash
NADA_SEARCH_BACKEND=opensearch
NADA_OPENSEARCH_URL=https://localhost:9200
NADA_OPENSEARCH_USER=nada_ai
NADA_OPENSEARCH_PASSWORD=<nada-ai-password>
NADA_OPENSEARCH_CA_CERTS=/etc/nada-ai/opensearch-root-ca.pem
NADA_OPENSEARCH_VERIFY_CERTS=true
NADA_INDEX_NAME=nada-metadata

NADA_EMBEDDING_BACKEND=local
NADA_EMBEDDING_MODEL_ID=microsoft/harrier-oss-v1-270m   # or the local folder
NADA_EMBEDDING_DEVICE=cpu

NADA_ADMIN_API_KEY=<openssl rand -hex 32>
AI4DATA_METADATA_CATALOG_URL=https://nada.example.org/index.php
AI4DATA_METADATA_CATALOG_X_API_KEY=<NADA admin API key>
AI4DATA_DISCOVERY_DATA_PATH=./data/nada-discovery
```

A NADA on the same server is reached at its normal URL (`http://localhost/...` works here, unlike in Docker).

---

## 6. Run the API as a service

`/etc/systemd/system/nada-ai.service`:

```ini
[Unit]
Description=nada-ai API (uvicorn)
After=network-online.target opensearch.service
Wants=opensearch.service

[Service]
Type=simple
User=nada-ai
WorkingDirectory=/opt/nada-ai
Environment=UV_CACHE_DIR=/var/cache/uv
Environment=HF_HOME=/var/cache/huggingface
Environment=HF_HUB_DISABLE_XET=1
ExecStart=/usr/local/bin/uv run --no-sync uvicorn nada_ai.app.main:app --host 127.0.0.1 --port 8020
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Keep a single uvicorn worker: the ingest gate and the job registry live in the process. Drop the
`After`/`Wants` on `opensearch.service` if OpenSearch is on another host.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nada-ai
sudo systemctl status nada-ai
curl -s http://127.0.0.1:8020/health
curl -s -X POST http://127.0.0.1:8020/health/embeddings/warmup -H "X-NADA-Admin-Key: $NADA_ADMIN_API_KEY"
journalctl -u nada-ai -f      # logs
```

---

## 7. Reverse proxy

Expose the API to NADA and other clients through Apache or nginx with HTTPS, and keep uvicorn on `127.0.0.1`.

### Apache

Needs `mod_proxy` and `mod_proxy_http` (loaded by default on RHEL; on Debian/Ubuntu run
`sudo a2enmod proxy proxy_http ssl`). A vhost, e.g. `/etc/httpd/conf.d/nada-ai.conf`:

```apache
<VirtualHost *:443>
    ServerName nada-ai.example.org
    SSLEngine on
    # SSLCertificateFile / SSLCertificateKeyFile ...

    ProxyPreserveHost On
    ProxyRequests Off
    ProxyTimeout 300
    ProxyPass        / http://127.0.0.1:8020/
    ProxyPassReverse / http://127.0.0.1:8020/

    ErrorLog  /var/log/httpd/nada-ai-error.log
    CustomLog /var/log/httpd/nada-ai-access.log combined
</VirtualHost>
```

If the server already hosts other sites, use its own `ServerName` (or a `<Location>` block) rather than replacing the
default site. On RHEL, SELinux blocks Apache from connecting to the API port until you allow it, and the firewall
needs HTTPS open:

```bash
sudo setsebool -P httpd_can_network_connect on
sudo firewall-cmd --permanent --add-service=https && sudo firewall-cmd --reload
sudo apachectl configtest && sudo systemctl reload httpd
curl -s https://nada-ai.example.org/health
```

### nginx

```nginx
server {
    listen 443 ssl;
    server_name nada-ai.example.org;
    # ssl_certificate / ssl_certificate_key ...

    location / {
        proxy_pass http://127.0.0.1:8020;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
```

If NADA runs on the same server, it can call `http://127.0.0.1:8020` directly and skip the proxy.

---

## 8. Next steps

Continue in the main guide:

1. [Create the indexes](./opensearch-setup-guide.md#5-create-the-indexes). Use `uv run` in place of `docker exec`.
2. [Index the catalog](./opensearch-setup-guide.md#6-index-the-catalog). Long CLI runs belong in `tmux` or `screen`;
   the HTTP jobs run inside the service.
3. [Keep the index in sync with NADA](./opensearch-setup-guide.md#7-keep-the-index-in-sync-with-nada).
4. [Connect NADA](./opensearch-setup-guide.md#8-connect-nada).
5. Production items: [replicas, snapshots, secrets](./opensearch-setup-guide.md#production-checklist).

To update nada-ai later:

```bash
cd /opt/nada-ai
sudo -u nada-ai git pull
sudo -u nada-ai env UV_CACHE_DIR=/var/cache/uv uv sync --extra local   # same flags as the first install
sudo systemctl restart nada-ai
```

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| OpenSearch will not start; `vm.max_map_count` in the log | Set it as in [Host settings](#host-settings) |
| `dnf`/`apt` install fails asking for a password | Pass `OPENSEARCH_INITIAL_ADMIN_PASSWORD` (a strong one; weak passwords are refused) |
| API logs `SSL: CERTIFICATE_VERIFY_FAILED` to OpenSearch | `NADA_OPENSEARCH_CA_CERTS` does not point at the CA that signed OpenSearch's certificate |
| `401` / `403` from OpenSearch | Wrong `nada_ai` password, or the role mapping is missing; re-run [step 3](#3-secure-opensearch-and-create-the-nada-ai-user) |
| `uv sync` fails with no space left on device | Caches on a small `/`. Set `UV_CACHE_DIR` and `HF_HOME` under `/var/cache` |
| Service starts, then the first search hangs | Model download blocked or slow. Pre-download it ([step 4](#get-the-harrier-model)) |
| Apache returns 503 on RHEL | SELinux: `sudo setsebool -P httpd_can_network_connect on` |

More: [OpenSearch guide troubleshooting](./opensearch-setup-guide.md#12-troubleshooting).
