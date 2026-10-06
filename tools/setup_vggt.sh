#!/bin/bash
# Clone VGGT into third_party/vggt at the commit used in our experiments and
# apply a one-line dtype fix to the DPT head (needed for bf16 point/depth heads).
# Run from the repo root.
set -e
mkdir -p third_party
if [ ! -d third_party/vggt ]; then
    git clone https://github.com/facebookresearch/vggt.git third_party/vggt
fi
git -C third_party/vggt checkout 44b3afbd1869d8bde4894dd8ea1e293112dd5eba
git -C third_party/vggt apply ../../patches/vggt_dpt_head_dtype.patch
