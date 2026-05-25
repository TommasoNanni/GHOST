#!/bin/bash
set -euo pipefail
ln -sf $CONDA_PREFIX/share/eigen3 $CONDA_PREFIX/share/Eigen3
ln -sf $CONDA_PREFIX/targets/sbsa-linux/lib/libnvToolsExt.so.1 $CONDA_PREFIX/targets/sbsa-linux/lib/libnvToolsExt.so
echo '#pragma once' > $CONDA_PREFIX/targets/sbsa-linux/include/nvToolsExt.h

CMAKE_ARGS="$CMAKE_ARGS \
    -DMOMENTUM_BUILD_RENDERER=OFF \
    -DMOMENTUM_BUILD_TESTING=OFF \
    -DMOMENTUM_BUILD_RASTERIZER=OFF \
    -DMOMENTUM_USE_SYSTEM_RERUN_CPP_SDK=ON \
    -Drerun_sdk_DIR=/tmp/rerun_sdk_install/lib/cmake/rerun_sdk \
    -DArrow_DIR=/tmp/arrow_install/lib/cmake/Arrow \
    -Ddrjit_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/drjit/cmake/drjit \
    -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/sbsa-linux \
    -DCUDAToolkit_ROOT=$CONDA_PREFIX/targets/sbsa-linux" \
CXXFLAGS="$CXXFLAGS -fsigned-char" \
python -m pip install --no-build-isolation --no-cache-dir \
    'pymomentum @ git+https://github.com/TommasoNanni/momentum.git@aarch64-no-rasterizer'

PYMOMENTUM_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/pymomentum
for f in $PYMOMENTUM_DIR/*.so; do
    patchelf --set-rpath '$ORIGIN/../torch/lib:'"$CONDA_PREFIX/lib" "$f" 2>/dev/null
done
echo 'pymomentum installed and patched'
