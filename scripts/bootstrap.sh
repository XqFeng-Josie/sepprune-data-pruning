#!/usr/bin/env bash
# Fetch the three upstream model implementations this project builds on.
#
# They are cloned at pinned commits rather than vendored: all three carry their
# own licences (MIT for AFRCNN and sudo_rm_rf, Apache-2.0 for TDANet) and
# redistributing them here would both duplicate that code and obscure which
# revision the results came from.
#
#   bash scripts/bootstrap.sh
#
# TIGER is deliberately absent: no code in this repository imports it.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${root}/third_party"

# name | url | pinned commit | what depends on it
pin() {
  local name="$1" url="$2" commit="$3" dir="${root}/third_party/$1"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone "${url}" "${dir}"
  fi
  git -C "${dir}" fetch --quiet origin "${commit}" 2>/dev/null || git -C "${dir}" fetch --quiet origin
  git -C "${dir}" checkout --quiet --detach "${commit}"
  printf '  %-12s %s\n' "${name}" "$(git -C "${dir}" rev-parse --short HEAD)"
}

echo "pinning third-party sources:"
# TDANet also provides the `look2hear` package that reproduction/tdanet_seprune.py imports.
pin TDANet     https://github.com/JusperLee/TDANet.git                    565af18692e18bf695e5bb0ca54ba466c4a86a2a
pin AFRCNN     https://github.com/JusperLee/AFRCNN-For-Speech-Separation.git 5ce11eb08fbb4f6c3d5013e6648b96ded999ea20
pin sudo_rm_rf https://github.com/etzinis/sudo_rm_rf.git                  cd00f2e21f7ad6281360cdf24ade36f84b0fbad6

cat <<'NOTE'

third_party is ready. Next:
  python -m pip install -r requirements.txt
  python scripts/download_lrs2mix.py      # 16.5 GB LRS2-2Mix archive
  python -m reproduction.data_pruning.verify --resume-test
NOTE
