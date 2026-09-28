#!/usr/bin/env bash
# Reproducible, resumable, parallel download of the PTB-XL 100 Hz signal corpus.
#
#   backend/dataset/download_ptbxl_signals.sh [--list F] [--root D] [--jobs N]
#                                             [--batch N] [--limit N]
#
# Reads record stems (one per line, e.g. "records100/00000/00001_lr") from the
# list file and fetches <stem>.hea + <stem>.dat from PhysioNet into the mirrored
# layout under the dataset root, so PTB-XL's own ``filename_lr`` paths resolve.
#
# Properties that matter for a multi-hour fetch over a slow link:
#
#   * Idempotent / resumable -- any file already on disk and non-empty is
#     skipped, so the script can be interrupted and re-run at will. It is also
#     the repair step: `ptbxl.py --verify-signals --delete-corrupt` removes any
#     file whose SHA-256 disagrees with PhysioNet's manifest, and re-running this
#     script refetches exactly those.
#   * CRLF-safe -- a CR in the list file once produced 398 silent
#     `curl: (3) URL rejected` failures, so every line is stripped explicitly.
#   * Parallel with connection reuse -- one curl process per batch with
#     `--parallel --parallel-max N` keeps N connections warm instead of paying a
#     TLS handshake per file. Measured on this link: sequential ~19 KB/s,
#     4 conns ~54 KB/s, 8 conns ~82 KB/s on byte throughput; for PTB-XL's ~12 KB
#     files the binding cost is per-request latency, and the measured file rate
#     was ~12.5 files/s at 8 and ~8 files/s at 16 under load. 8 is the default:
#     it was the faster of the two in the sustained run and it is the politer
#     one against a public research host.
#   * Circuit breaker -- a batch that fetches nothing is treated as a link
#     failure, not as 600 absent records. The first bulk run lost 13,708 files
#     to a local DNS outage ("Could not resolve host" x53,608) and raced through
#     the remainder at 0 files/s; now such a batch is retried after a pause and
#     the run aborts rather than reporting an outage as missing data.
#   * Honest tally -- prints per-batch progress plus a final present/expected
#     count, and exits non-zero if anything is still missing. Downloading is NOT
#     verification: run the SHA-256 verifier afterwards.
#
# Deliberate scope note: the default list covers strat_folds 1-9 only. Fold 10 is
# the frozen test partition and is intentionally never fetched during
# development, so the guarantee that it was not inspected is structural (the
# bytes are not on the machine) rather than a promise.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${REPO_ROOT}/backend/artifacts/dataset/ptbxl"
BASE_URL="https://physionet.org/files/ptb-xl/1.0.3"
LIST=""
JOBS=8
BATCH=600
LIMIT=0
STALL_RETRIES=3      # attempts at a batch that fetched nothing
STALL_SLEEP=45       # seconds to wait between those attempts
MAX_DEAD_BATCHES=2   # consecutive dead batches before giving up

while [ $# -gt 0 ]; do
  case "$1" in
    --list)  LIST="$2";  shift 2 ;;
    --root)  ROOT="$2";  shift 2 ;;
    --jobs)  JOBS="$2";  shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "${LIST}" ] || LIST="${ROOT}/dev_records.txt"

if [ ! -f "${LIST}" ]; then
  echo "stem list not found: ${LIST}" >&2
  exit 2
fi

WORK="${ROOT}/_fetch"
mkdir -p "${WORK}"
MISSING="${WORK}/missing.txt"
CFG="${WORK}/batch.cfg"
PRESENT="${WORK}/present.txt"

# curl here is a native Windows binary, which does not understand Git Bash's
# "/c/Users/..." view of the filesystem: it would write to C:\c\Users\...
# Everything bash touches keeps the POSIX path; only curl's output= is translated.
if command -v cygpath >/dev/null 2>&1; then
  CURL_ROOT="$(cygpath -m "${ROOT}")"
else
  CURL_ROOT="${ROOT}"
fi

echo "PTB-XL signal fetch"
echo "  root ...... ${ROOT}"
echo "  list ...... ${LIST}"
echo "  jobs ...... ${JOBS}   batch ...... ${BATCH}   limit ...... ${LIMIT}"

# ---------------------------------------------------------------------------
# What is already on disk, and therefore what is missing.
#
# One find + one awk pass rather than two stat() calls per record: a bash loop
# over 19,601 records costs ~20 minutes of pure interpreter time on Windows,
# which would be paid twice (before and after the fetch).
# ---------------------------------------------------------------------------
scan_missing() {
  {
    find "${ROOT}/records100" -type f -size +0c -printf 'records100/%P\n' 2>/dev/null
    find "${ROOT}/records500" -type f -size +0c -printf 'records500/%P\n' 2>/dev/null
  } > "${PRESENT}"

  awk -v limit="${LIMIT}" -v out="${MISSING}" '
    NR == FNR { have[$0] = 1; next }
    {
      line = $0
      gsub(/[[:space:]]/, "", line)          # also removes any CR
      if (line == "" || line ~ /^#/) next
      if (limit > 0 && nrec >= limit) next
      nrec++
      split("hea dat", ext, " ")
      for (i = 1; i <= 2; i++) {
        file = line "." ext[i]
        expected++
        if (file in have) present++
        else { missing++; print file > out }
      }
    }
    END {
      close(out)
      printf "%d %d %d %d\n", nrec+0, expected+0, present+0, missing+0
    }
  ' "${PRESENT}" "${LIST}"
}

# Zero-byte files are stubs from an aborted run; the find above already ignores
# them, but remove them so a retry can write cleanly.
stubs="$(find "${ROOT}/records100" "${ROOT}/records500" -type f -size 0 -print -delete 2>/dev/null | wc -l)"
[ "${stubs}" -gt 0 ] && echo "  removed ${stubs} zero-byte stub(s) from a previous run"

: > "${MISSING}"
read -r records expected present missing <<< "$(scan_missing)"
echo "  records ... ${records}"
echo "  files ..... ${expected} expected, ${present} already present, ${missing} to fetch"

if [ "${missing}" -eq 0 ]; then
  echo "DONE: nothing to fetch; all ${expected} files present"
  exit 0
fi

# ---------------------------------------------------------------------------
# Fetch in batches. One curl process per batch, N transfers in flight, so
# connections are reused across the batch instead of per file.
# ---------------------------------------------------------------------------
start=${SECONDS}
fetched=0
failed=0
batch_no=0
dead_batches=0
batch_got=0
batch_miss=0
aborted=0
n_batches=$(( (missing + BATCH - 1) / BATCH ))

run_batch_once() {
  : > "${CFG}"
  while IFS= read -r rel; do
    [ -z "${rel}" ] && continue
    printf 'url = "%s/%s"\noutput = "%s/%s"\n' "${BASE_URL}" "${rel}" "${CURL_ROOT}" "${rel}" >> "${CFG}"
  done < "${WORK}/batch.list"

  curl --parallel --parallel-max "${JOBS}" \
       --fail --silent --show-error \
       --create-dirs --remove-on-error \
       --retry 3 --retry-delay 2 --retry-connrefused \
       --connect-timeout 30 --max-time 300 \
       -K "${CFG}" 2>> "${WORK}/curl.err"

  # Trust the filesystem, not curl's exit code: count what actually landed.
  batch_got=0
  batch_miss=0
  while IFS= read -r rel; do
    [ -z "${rel}" ] && continue
    if [ -s "${ROOT}/${rel}" ]; then
      batch_got=$((batch_got + 1))
    else
      batch_miss=$((batch_miss + 1))
    fi
  done < "${WORK}/batch.list"
}

run_batch() {
  # Circuit breaker. A batch that fetches *nothing* means the link is down, not
  # that 600 records are absent upstream: the first bulk run hit a local DNS
  # outage ("Could not resolve host" x53,608) and cheerfully raced through the
  # remaining 13,708 files at 0 files/s, marking them all missing. Pause and
  # retry the same batch instead, and give up entirely rather than write a run
  # whose "missing" column is really an outage.
  local attempt=1
  while : ; do
    run_batch_once
    if [ "${batch_got}" -gt 0 ] || [ "${attempt}" -ge "${STALL_RETRIES}" ]; then
      break
    fi
    echo "  batch ${batch_no}: nothing fetched (attempt ${attempt}/${STALL_RETRIES}); pausing ${STALL_SLEEP}s -- network/DNS stall?"
    sleep "${STALL_SLEEP}"
    attempt=$((attempt + 1))
  done

  fetched=$((fetched + batch_got))
  failed=$((failed + batch_miss))
  if [ "${batch_got}" -eq 0 ]; then
    dead_batches=$((dead_batches + 1))
  else
    dead_batches=0
  fi
}

report_batch() {
  local elapsed=$(( SECONDS - start ))
  local done_files=$(( fetched + failed ))
  local rate_x10=0
  [ "${elapsed}" -gt 0 ] && rate_x10=$(( done_files * 10 / elapsed ))
  local eta="?"
  [ "${rate_x10}" -gt 0 ] && eta="$(( (missing - done_files) * 10 / rate_x10 ))s"
  echo "  batch ${batch_no}/${n_batches}: ${fetched} fetched, ${failed} missing, ${elapsed}s elapsed, ~${eta} left (${rate_x10}/10 files/s)"
}

: > "${WORK}/curl.err"
: > "${WORK}/batch.list"
lines_in_batch=0
while IFS= read -r rel || [ -n "${rel}" ]; do
  [ -z "${rel}" ] && continue
  printf '%s\n' "${rel}" >> "${WORK}/batch.list"
  lines_in_batch=$((lines_in_batch + 1))
  if [ "${lines_in_batch}" -ge "${BATCH}" ]; then
    batch_no=$((batch_no + 1))
    run_batch
    report_batch
    : > "${WORK}/batch.list"
    lines_in_batch=0
    if [ "${dead_batches}" -ge "${MAX_DEAD_BATCHES}" ]; then
      echo "ABORTING: ${dead_batches} consecutive batches fetched nothing after ${STALL_RETRIES} attempts each."
      echo "  The link, not the dataset, is the problem -- see ${WORK}/curl.err."
      echo "  Nothing is lost: re-run this script when the network is back and it resumes."
      aborted=1
      break
    fi
  fi
done < "${MISSING}"

if [ "${lines_in_batch}" -gt 0 ] && [ "${aborted}" -eq 0 ]; then
  batch_no=$((batch_no + 1))
  run_batch
  report_batch
fi

# ---------------------------------------------------------------------------
# Final tally, recomputed from disk rather than from the running counters.
# ---------------------------------------------------------------------------
: > "${MISSING}"
read -r f_records f_expected f_present f_missing <<< "$(scan_missing)"
cp "${MISSING}" "${WORK}/still_missing.txt"

echo "DONE: ${f_present}/${f_expected} files present for ${f_records} records, ${f_missing} missing, $(( SECONDS - start ))s"
if [ "${f_missing}" -gt 0 ]; then
  echo "still-missing list: ${WORK}/still_missing.txt (re-run this script to retry)"
  echo "curl errors: ${WORK}/curl.err"
fi
echo "NOTE: presence is not integrity. Verify with:"
echo "  python -m backend.dataset.ptbxl --verify-signals '${LIST}' --delete-corrupt"
[ "${f_missing}" -eq 0 ]
