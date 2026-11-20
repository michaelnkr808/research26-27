#!/usr/bin/env bash
# Download the gated VGGT-Omega file with the lead's Hugging Face token (RUN.md step 4).
# Run by the user in their own interactive SSH session on the machine. The token is read
# at a hidden prompt; it never appears on a command line, in shell history, in a file or
# in an agent transcript.
set -euo pipefail
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate wnm_3d
read -rsp "Hugging Face token (input hidden): " HF_TOKEN
echo
HF_TOKEN="$HF_TOKEN" hf download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --revision 1041e80fc0e911235d3426b0a3d9a81075111a53 --local-dir /workspace/WNM-3D/checkpoints
unset HF_TOKEN
HFH="${HF_HOME:-$HOME/.cache/huggingface}"
for f in "$HFH/token" "$HFH/stored_tokens"; do
  if [ -e "$f" ]; then echo "TOKEN FILE PRESENT: $f (delete it: rm -f $f)"; else echo "no $f"; fi
done
echo "done; close this session"
