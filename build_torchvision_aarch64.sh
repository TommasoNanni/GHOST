#!/bin/bash
set -euo pipefail
whldir=$(mktemp -d)
PATH=$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/bin:$PATH FORCE_CUDA=1 CUDA_HOME=$CONDA_PREFIX/targets/sbsa-linux \
    python -m pip wheel --no-build-isolation --no-cache-dir --no-deps -w $whldir \
    'git+https://github.com/pytorch/vision.git@v0.22.1'
unzip -o $whldir/torchvision-*.whl 'torchvision/*.so' -d $whldir/extracted
TV_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/torchvision
for f in $whldir/extracted/torchvision/*.so; do
    cp --remove-destination $f $TV_DIR/
done
for f in $TV_DIR/_C.so $TV_DIR/image.so $TV_DIR/video_reader.so; do
    [ -f $f ] && patchelf --set-rpath '$ORIGIN/../torch/lib:'"$CONDA_PREFIX/lib" $f 2>/dev/null
done
echo 'torchvision CUDA .so files installed and patched'
