# Deploying the Meme Retrieval Demo

The deployment is a FastAPI app serving seven pages (`/`, `/clip`, `/clip-nlp`, `/vlm`,
`/compare`, `/evaluation`, `/settings`) plus a small JSON API. All state lives in `./data`, which
is mounted into the container.

Two exposure options:

- **A. Direct IP** — quickest, `http://<SERVER_IP>:8000`.
- **B. Domain + HTTPS** — Caddy issues and renews certificates automatically.

New to the project? Complete the build steps in [README.md](README.md) first; the
server needs a finished `data/` directory, not source-only.

---

## 1. Get the data

### Prebuilt (recommended)

The [`v1-data` release](https://github.com/yuchokwong/Poly-COMP-V2/releases/tag/v1-data)
contains a finished `data/` directory — 6,378 images, the SQLite gallery, all three
indexes and the evaluation reports — so the server needs neither network access to
HuggingFace nor a GPU:

```bash
cd ~/meme-retrieval
curl -L https://github.com/yuchokwong/Poly-COMP-V2/releases/download/v1-data/data.tar.gz | tar -xz
```

That completes this section; skip to section 2. The archive is ~540 MB and extracts in
place as `data/`. The first search still downloads the CLIP and BGE weights (~2 GB) into
the `openclip_cache` volume.

### Build it yourself (once)

Only needed when you change the dataset, the CLIP backbone or the VLM prompt. Build it on
a machine with network access:
python -m tsr prepare                     # ~1.7 GB download, 6,382 images
python -m tsr build --strategy gallery     # SQLite rows + shared CLIP index
python -m tsr build --strategy clip_nlp    # annotation docs + BGE index
python -m tsr make-eval --num 60           # small demo benchmark
python -m tsr compare                      # reports/compare.{json,html} + failures.html
```

For strategy C, caption on a GPU host (native install, not the serving image):

```bash
# install the CUDA torch wheel for your driver first
pip install -r requirements-vlm.txt
python -m tsr vlm-caption --limit 20 --out data/vlm_sample.html   # check quality
TSR_MODELS__VLM__BATCH_SIZE=8 python -m tsr vlm-caption            # resumable
# batch 8 measured ~1.5 s/img on a 12 GB RTX 4070 SUPER; drop to 4 if it OOMs
# (a failed batch automatically falls back to one image at a time)
python -m tsr build --strategy vlm
```

Then copy the results to the server, or upload them once as a release asset and
download it there with `curl`:

```bash
rsync -av data/ user@server:~/meme-retrieval/data/
```

Do **not** copy `data/.cache/` (thumbnails are regenerated), `data/raw/` (only needed
to re-run `prepare`) or the SQLite `-wal`/`-shm` side files. To caption on the server
instead, install the VLM extras natively there and run the same commands — the base
Docker image deliberately excludes `transformers`' VLM stack extras.

Approximate sizes after a full build:

| Path | Size | Notes |
| --- | --- | --- |
| `data/images/` | 0.6–1.2 GB | 6,382 JPEGs, max side 768 |
| `data/gallery.db` | tens of MB | SQLite v2 + FTS5 |
| `data/index/clip/` | ~20 MB | 6,382 × 768-d float32 |
| `data/index/clip_nlp/` | ~10 MB | 6,382 × 384-d BGE vectors |
| `data/index/vlm/` | ~10 MB | BGE vectors over VLM docs |
| `data/vlm/descriptions.jsonl` | tens of MB | resumable VLM cache |
| `data/eval/`, `data/reports/` | small | queries, qrels, reports |

The first search downloads model weights into the `openclip_cache` volume
(OpenCLIP ViT-L-14 is ~1.7 GB; BGE-small is ~130 MB; the optional reranker ~1.1 GB).

Verify locally before shipping:

```bash
python -m tsr stats --json
python -m tsr serve
curl http://localhost:8000/api/health
```

`status: ok` means at least one strategy is ready; `degraded` with
`store_error`/per-strategy errors names what is missing.

---

## A. Direct IP (no domain)

### On the Ubuntu server

```bash
# 1. Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker

# 2. Put the project and the built data/ in place
git clone https://github.com/yuchokwong/Poly-COMP-V2.git ~/meme-retrieval
cd ~/meme-retrieval
# then copy the data/ built in section 1, e.g. from the build machine:
#   rsync -av data/ user@server:~/meme-retrieval/data/

# 3. Open the app port (also in the cloud security group, if any)
sudo ufw allow 8000/tcp

# 4. Start
docker compose up -d --build

# 5. Find the public IP
curl -s ifconfig.me; echo
```

Others open `http://<SERVER_IP>:8000`. Verify:

```bash
curl http://<SERVER_IP>:8000/api/health
```

To serve on port 80 instead, put `TSR_APP__PORT=80` in `.env`,
`sudo ufw allow 80/tcp`, and `docker compose up -d`.

> A plain IP cannot have HTTPS; certificates require a domain.

---

## B. Domain + HTTPS

1. Create an **A record** for e.g. `search.example.com` pointing at the public IP.
   Verify with `dig +short search.example.com`.
2. Open ports 80 and 443 (`sudo ufw allow 80/tcp; sudo ufw allow 443/tcp`, and in
   the cloud security group).
3. Configure and start:

```bash
cd ~/meme-retrieval
cp .env.example .env
# edit .env -> DOMAIN=search.example.com and ACME_EMAIL=you@example.com
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml logs -f caddy
```

Caddy obtains and renews TLS automatically. The app is exposed only on the internal
network (`expose: 8000`), never directly to the host.

> No domain but want HTTPS? Use `<SERVER_IP>.sslip.io` as `DOMAIN`; it resolves to
> that IP.

---

## Environment

`.env` is optional and is read by every compose file. The values most worth setting:

| Variable | Purpose |
| --- | --- |
| `TSR_APP__PORT` | Host port for the base compose file (default 8000) |
| `TSR_APP__PRELOAD_MODELS` | Load all strategies at startup (compose default `true`); set `false` to load on first request |
| `OMP_NUM_THREADS` | Torch thread cap; defaults to the container's CPU quota/affinity |
| `TSR_MODELS__CLIP__DEVICE` | `cpu` or `cuda` (use `docker-compose.gpu.yml`) |
| `TSR_MODELS__CLIP__CACHE_DIR` | Model cache directory; set by compose to `/root/.cache/open_clip` |
| `TSR_MODELS__TEXT__DEVICE` | Device for BGE query encoding |
| `TSR_MODELS__VLM__BACKEND` | `qwen2_5_vl` or `none` |
| `TSR_STRATEGIES__CLIP_NLP__WEIGHTS__*` | Runtime-overridable fusion weights |
| `TSR_STRATEGIES__CLIP_NLP__RERANK__ENABLED` | Turn the cross-encoder on/off |
| `TSR_CONFIG` | Extra YAML layered over `configs/default.yaml` |
| `DOMAIN`, `ACME_EMAIL` | Only for `docker-compose.prod.yml` |
| `TORCH_INDEX_URL` | Build-time torch wheels (`cpu`, `cu121`) |

Nested configuration uses `__`; full defaults live in `configs/default.yaml`.

### GPU

The serving container can use CUDA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

This sets `TSR_MODELS__*__DEVICE=cuda` and reserves all GPUs. It does not install the
VLM extras; run `vlm-caption` outside the container (or build a custom image with
`requirements-vlm.txt`).

---

## Updating

```bash
cd ~/meme-retrieval
git pull
docker compose up -d --build          # or -f docker-compose.prod.yml
```

`data/` is gitignored and untouched. Rebuild indexes only when the dataset, model,
or document text changes:

```bash
docker compose --profile tools run --rm build python -m tsr build --strategy all
docker compose restart app
```

Restart the API after rebuilding: readiness and stats are a load-time snapshot, so a
running process keeps serving the indexes it loaded at startup (it will not pick up a
new index, and `stats` deliberately reports the loaded artifacts rather than newer
files on disk).

Configuration changes that affect only ranking weights can be applied from the
`/settings` page (or `GET`/`PUT /api/settings`; persisted in `data/settings.json`)
or by restarting with a new `.env`. Edits apply from the next search; models, index
type and dataset remain build-time choices.

## Operations

```bash
docker compose ps
docker compose logs -f app
docker compose restart app
docker compose down                   # data/ on the host is untouched
docker compose --profile tools run --rm build python -m tsr stats --json
```

## Native install (no Docker)

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
cd ~/meme-retrieval
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
python -m tsr serve --host 0.0.0.0 --port 8000
```

Behind nginx + certbot:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
# proxy_pass http://127.0.0.1:8000;  then:
sudo certbot --nginx -d search.example.com
```

## Troubleshooting

- **Cannot connect** — the port is closed; check both `ufw` and the cloud security
  group. Test on the host with `curl http://localhost:8000/api/health`.
- **`ready: false` / `status: degraded`** — read `strategies[].error` and
  `store_error` in `/api/health`. Typical causes: `gallery.db` missing (run
  `prepare` + `build --strategy gallery`), CLIP index missing, or VLM docs never
  generated. `docker compose logs app` shows the same details.
- **`SchemaVersionError` on startup** — `data/gallery.db` comes from the old COCO
  schema. Delete `data/gallery.db` (or the whole `data/` directory) and rebuild;
  there is no in-place migration.
- **FAISS id-map errors** — `image_ids.npy` is missing or its length does not match
  the index. Rebuild the strategy index; do not mix indexes from different builds.
- **Strategy C unavailable** — `models.vlm.backend: none` or `data/index/vlm/` was
  never built. The other strategies and `/compare` still work.
- **First search is slow** — model weights are downloading. Check
  `docker compose logs -f app`; subsequent starts reuse `openclip_cache`.
- **Out of memory on CPU** — lower `TSR_APP__DEFAULT_TOP_K`, set
  `TSR_STRATEGIES__CLIP_NLP__RERANK__ENABLED=false`, or use a smaller
  `TSR_MODELS__CLIP__NAME` (e.g. `ViT-B-32`) and rebuild the image index.
- **Dynamic public IP** — IP-based URLs break when the address changes; use a static
  IP or a domain.
