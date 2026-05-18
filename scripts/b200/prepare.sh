#!/usr/bin/env bash
# B200 server one-shot data-prep script.
#
# Assumes:
#   - Repo checked out at /workspace/code/restora-models
#   - `uv sync` already ran (the synthesiser uses cv2/numpy from the venv)
#
# What it does, in order:
#   1. Creates /workspace/data/{reds,film-overlays}
#   2. Downloads REDS train_sharp + val_sharp from the official
#      Hugging Face mirror (snah/REDS) via aria2c (or wget as fallback).
#      Skips already-downloaded zips. Override the URLs via
#      $REDS_TRAIN_SHARP_URLS / $REDS_VAL_SHARP_URLS env vars.
#   3. Unpacks zips into the expected `<root>/<split>/<seq>/<frame>.png`
#      layout. Skips splits that already have sequences.
#   4. Synthesises 600 film-overlay PNGs via the existing CLI.
#   5. Verifies layout and prints a summary.
#
# Re-runnable: every step short-circuits if its output already exists.

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths — fixed for the B200 deployment.
# -----------------------------------------------------------------------------
REPO_DIR="/workspace/code/restora-models"
DATA_DIR="/workspace/data"
REDS_DIR="${DATA_DIR}/reds"
OVERLAY_DIR="${DATA_DIR}/film-overlays"
ZIP_CACHE="${DATA_DIR}/_zips"

# -----------------------------------------------------------------------------
# REDS download URLs — Hugging Face mirror officially listed on
# https://seungjunnah.github.io/Datasets/reds.html (alongside Google
# Drive). HF resolves these as plain HTTPS, no auth, fully resumable.
# Override on the command line if the URLs ever change:
#     REDS_TRAIN_SHARP_URLS="..." REDS_VAL_SHARP_URLS="..." ./prepare.sh
# Space-separated if a split is delivered in multiple parts.
# -----------------------------------------------------------------------------
: "${REDS_TRAIN_SHARP_URLS:=https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip}"
: "${REDS_VAL_SHARP_URLS:=https://huggingface.co/datasets/snah/REDS/resolve/main/val_sharp.zip}"

# Colour helpers (no-op if not a TTY).
if [ -t 1 ]; then
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_GREEN=""; C_YELLOW=""; C_RED=""; C_BOLD=""; C_OFF=""
fi
log()    { printf '%s[prep]%s %s\n' "${C_BOLD}" "${C_OFF}" "$*"; }
warn()   { printf '%s[prep]%s %s%s%s\n' "${C_BOLD}" "${C_OFF}" "${C_YELLOW}" "$*" "${C_OFF}"; }
err()    { printf '%s[prep]%s %s%s%s\n' "${C_BOLD}" "${C_OFF}" "${C_RED}"    "$*" "${C_OFF}" >&2; }
ok()     { printf '%s[prep]%s %s%s%s\n' "${C_BOLD}" "${C_OFF}" "${C_GREEN}"  "$*" "${C_OFF}"; }

# -----------------------------------------------------------------------------
# Pre-flight.
# -----------------------------------------------------------------------------
log "preflight: repo=${REPO_DIR} data=${DATA_DIR}"
if [ ! -d "${REPO_DIR}" ]; then
  err "expected repo at ${REPO_DIR} — clone it there first"
  exit 1
fi
if [ ! -f "${REPO_DIR}/pyproject.toml" ]; then
  err "${REPO_DIR} doesn't look like the restora-models repo (no pyproject.toml)"
  exit 1
fi

mkdir -p "${REDS_DIR}" "${OVERLAY_DIR}" "${ZIP_CACHE}"

# Pick a downloader. aria2c is ~3-5x faster for multi-part downloads
# because it pipelines connections; wget is the universal fallback.
DOWNLOADER=""
if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER="aria2c"
  log "using aria2c for downloads"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
  log "using wget for downloads (install aria2c for faster multi-part fetch)"
else
  err "need aria2c or wget on PATH"
  exit 1
fi

# Need `unzip` for the REDS archives.
if ! command -v unzip >/dev/null 2>&1; then
  err "need unzip on PATH (apt install unzip)"
  exit 1
fi

# -----------------------------------------------------------------------------
# Download + unpack one REDS split.
# -----------------------------------------------------------------------------
#   $1 = split name (train_sharp / val_sharp)
#   $2 = space-separated list of URLs
download_split() {
  local split="$1"
  local urls="$2"
  local target_dir="${REDS_DIR}/${split}"

  # Already populated? Skip the whole thing.
  if [ -d "${target_dir}" ] && [ -n "$(ls -A "${target_dir}" 2>/dev/null)" ]; then
    local n_seqs
    n_seqs=$(find "${target_dir}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    ok "${split}: already populated (${n_seqs} sequences at ${target_dir}); skipping"
    return 0
  fi

  if [ -z "${urls// /}" ]; then
    warn "${split}: no URLs set — pass REDS_${split^^}_URLS to enable download"
    return 0
  fi

  mkdir -p "${target_dir}"
  local zip_subdir="${ZIP_CACHE}/${split}"
  mkdir -p "${zip_subdir}"

  # Download every URL into the per-split zip cache. -c resumes partial
  # files, so re-running the script after a network drop just continues.
  # The aria2c control file (`<out>.aria2`) marks an in-progress download
  # and is removed on completion — its absence is the canonical "done" flag.
  for url in ${urls}; do
    local fname
    fname=$(basename "${url%%\?*}")
    local out="${zip_subdir}/${fname}"
    if [ -f "${out}" ] && [ -s "${out}" ] && [ ! -f "${out}.aria2" ]; then
      local size
      size=$(du -h "${out}" | awk '{print $1}')
      ok "${split}: ${fname} already complete in cache (${size}) — skipping fetch"
      continue
    fi
    if [ -f "${out}.aria2" ]; then
      warn "${split}: ${fname} has a stale .aria2 control file — resuming"
    fi
    log "${split}: downloading ${fname}"
    case "${DOWNLOADER}" in
      aria2c)
        # -x 8: 8 connections per file. -s 8: 8 split download. --summary-interval
        # prints throughput every 5 s so you can watch progress.
        aria2c -x 8 -s 8 --summary-interval=5 --continue=true \
               --dir="${zip_subdir}" --out="${fname}" "${url}"
        ;;
      wget)
        wget --continue --show-progress --progress=dot:giga \
             -O "${out}" "${url}"
        ;;
    esac
  done

  # Unpack each archive. REDS zips usually contain a top-level
  # `train_sharp/...` or `val_sharp/...` dir. We unzip into ${REDS_DIR}
  # so the path becomes ${REDS_DIR}/${split}/<seq>/<frame>.png either
  # way (zip's own dirstructure or our explicit fallback).
  #
  # Progress + speed: serial `unzip` on a 30 GB / 24k-file REDS zip is
  # ~10-20 minutes and silent. We parallelize by feeding the central
  # directory file list (`unzip -Z1`) into `xargs -P N`, where each
  # worker opens the zip and extracts a chunk of files concurrently.
  # PNG payloads are already compressed (zip uses STORE), so the
  # bottleneck is per-file open/write syscall overhead — exactly the
  # workload parallelism helps.
  #
  # A background watcher prints `du -sh ${REDS_DIR}` every 5 s. That
  # sidesteps the stdio-buffering quirk that makes awk-on-unzip-pipe
  # appear hung (glibc fully-buffers stdout when it's a pipe), and it
  # works regardless of how many xargs workers are running.
  local NCPU
  NCPU=$(nproc 2>/dev/null || echo 4)
  # Past ~8 workers I/O contention erases the gains on most filesystems.
  [ "${NCPU}" -gt 8 ] && NCPU=8

  for zip in "${zip_subdir}"/*.zip; do
    [ -f "${zip}" ] || continue
    local total zip_size start_ts watcher_pid extract_rc=0
    total=$(unzip -Z1 "${zip}" | wc -l)
    zip_size=$(du -h "${zip}" | awk '{print $1}')
    log "${split}: unpacking $(basename "${zip}") — ${total} entries, ${zip_size}, ${NCPU}-way parallel"

    start_ts=$(date +%s)
    (
      while true; do
        sleep 5
        local elapsed sz
        elapsed=$(( $(date +%s) - start_ts ))
        sz=$(du -sh "${REDS_DIR}" 2>/dev/null | awk '{print $1}')
        printf '  [extract] %s unpacked — %dm%02ds elapsed\n' "${sz:-?}" "$((elapsed/60))" "$((elapsed%60))"
      done
    ) &
    watcher_pid=$!

    # Phase 1: pre-create every directory the zip will produce. Parallel
    # unzip workers race on the lazy mkdir() and one of them loses on
    # EEXIST — silently dropping a file. Pre-creating up front (single
    # idempotent `mkdir -p` call with every ancestor path) eliminates
    # the race. Awk emits every ancestor of every entry so even deeply
    # nested zips are covered.
    unzip -Z1 "${zip}" \
      | awk -F/ '{
          path = ""
          for (i = 1; i < NF; i++) {
            path = (i == 1) ? $1 : path "/" $i
            print path
          }
        }' \
      | sort -u \
      | (cd "${REDS_DIR}" && xargs -d '\n' -r mkdir -p)

    # Phase 2: parallel file extraction. Skip pure directory entries
    # (those ending in `/`) — they're already created by Phase 1, and
    # passing them to `unzip -oq` would just be wasted work.
    unzip -Z1 "${zip}" \
      | grep -v '/$' \
      | xargs -P "${NCPU}" -n 200 unzip -oq "${zip}" -d "${REDS_DIR}" \
      || extract_rc=$?

    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true

    if [ "${extract_rc}" -ne 0 ]; then
      err "${split}: extraction failed with exit code ${extract_rc}"
      return "${extract_rc}"
    fi

    local final_elapsed
    final_elapsed=$(( $(date +%s) - start_ts ))
    ok "${split}: unpacked $(basename "${zip}") in $((final_elapsed/60))m$((final_elapsed%60))s"
  done

  # If the archives didn't contain a `${split}/` top dir but dropped
  # sequences at REDS_DIR root, move them under ${split}/.
  if [ ! -d "${target_dir}" ] || [ -z "$(ls -A "${target_dir}" 2>/dev/null)" ]; then
    log "${split}: archives unpacked flat; consolidating into ${target_dir}"
    mkdir -p "${target_dir}"
    find "${REDS_DIR}" -mindepth 1 -maxdepth 1 -type d \
         ! -name "${split}" ! -name "train_sharp" ! -name "val_sharp" \
         -exec mv {} "${target_dir}/" \;
  fi

  local n_seqs
  n_seqs=$(find "${target_dir}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  ok "${split}: ${n_seqs} sequences ready at ${target_dir}"
}

# -----------------------------------------------------------------------------
# Step 1+2: REDS download + unpack.
# -----------------------------------------------------------------------------
log "==== step 1/3: REDS train_sharp ===="
download_split "train_sharp" "${REDS_TRAIN_SHARP_URLS}"

log "==== step 2/3: REDS val_sharp ===="
download_split "val_sharp"   "${REDS_VAL_SHARP_URLS}"

# -----------------------------------------------------------------------------
# Step 3: film overlay textures (synthetic — DeepRemaster source is dead).
# -----------------------------------------------------------------------------
log "==== step 3/3: film overlays ===="
n_existing=$(find "${OVERLAY_DIR}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
if [ "${n_existing}" -ge 600 ]; then
  ok "film-overlays: already have ${n_existing} PNGs at ${OVERLAY_DIR}; skipping"
else
  log "film-overlays: synthesising 600 textures (~50 MB)"
  cd "${REPO_DIR}"
  uv run restora prepare-data film-overlays \
      --out "${OVERLAY_DIR}" \
      --synthetic-only \
      --n-synthetic 600
fi

# -----------------------------------------------------------------------------
# Final summary + layout verification.
# -----------------------------------------------------------------------------
echo
log "==== summary ===="
for split in train_sharp val_sharp; do
  if [ -d "${REDS_DIR}/${split}" ]; then
    n=$(find "${REDS_DIR}/${split}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    frames=$(find "${REDS_DIR}/${split}" -name '*.png' 2>/dev/null | wc -l)
    ok "REDS ${split}: ${n} sequences, ${frames} frames"
  else
    warn "REDS ${split}: NOT PRESENT"
  fi
done
n_overlays=$(find "${OVERLAY_DIR}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
ok "film-overlays: ${n_overlays} PNGs"

# Ask the existing CLI to validate the REDS layout (this is what the
# trainer's data builder will see).
if [ -d "${REDS_DIR}/train_sharp" ]; then
  log "verifying REDS layout via restora CLI..."
  cd "${REPO_DIR}"
  uv run restora prepare-data reds --out "${REDS_DIR}" || \
    warn "restora prepare-data reds reported issues — inspect output above"
fi

ok "prepare.sh done."
echo
log "next: see scripts/b200/RUNBOOK.md for tmux start commands"
