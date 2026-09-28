#!/usr/bin/env bash
# Reproducible, CRLF-safe download of a bounded PTB-XL 100 Hz signal sample.
# Reads record stems from sample_records.txt (one per line), strips any CR,
# and fetches <stem>.hea + <stem>.dat from PhysioNet into the mirrored layout
# under backend/artifacts/dataset/ptbxl/ so that filename_lr paths resolve.
# Idempotent: skips files already present. Prints a final present/attempted tally.
set -u

ROOT="C:/Users/ISHAN SHUKLA/Downloads/Orqis-main/Orqis-main/backend/artifacts/dataset/ptbxl"
BASE_URL="https://physionet.org/files/ptb-xl/1.0.3"
LIST="${ROOT}/sample_records.txt"

ok=0
attempted=0
files_present=0

fetch() {
  # $1 = relative path under the dataset root (also the URL suffix)
  local rel="$1"
  local dest="${ROOT}/${rel}"
  local url="${BASE_URL}/${rel}"
  if [ -s "${dest}" ]; then
    files_present=$((files_present + 1))
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  # up to 3 attempts, generous per-file timeout for the slow link
  local tries=0
  while [ ${tries} -lt 3 ]; do
    if curl -sS --fail --max-time 180 -o "${dest}" "${url}"; then
      if [ -s "${dest}" ]; then
        files_present=$((files_present + 1))
        return 0
      fi
    fi
    tries=$((tries + 1))
    sleep 1
  done
  rm -f "${dest}"  # remove any empty stub so a rerun retries cleanly
  echo "FAILED: ${rel}"
  return 1
}

while IFS= read -r raw || [ -n "${raw}" ]; do
  stem="$(printf '%s' "${raw}" | tr -d '\r' | tr -d '[:space:]')"
  [ -z "${stem}" ] && continue
  attempted=$((attempted + 1))
  fetch "${stem}.hea" && fetch "${stem}.dat" && ok=$((ok + 1))
done < "${LIST}"

echo "DONE: attempted ${attempted} records, ${ok} fully downloaded, ${files_present} files present"
find "${ROOT}/records100" -type f 2>/dev/null | wc -l
