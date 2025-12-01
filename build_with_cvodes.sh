#!/bin/bash
# Build script for dFUSE with CVODES wrapper
# The wrapper must be compiled WITHOUT Enzyme to avoid SUNDIALS conflicts

set -e

SUNDIALS_ROOT=${SUNDIALS_ROOT:-/Users/darrieythorsson/compHydro/data/SYMFLUENCE_data/installs/sundials/install/sundials}

echo "=== Building CVODES wrapper (without Enzyme) ==="
mkdir -p build

# Step 1: Compile cvodes_wrapper.cpp WITHOUT Enzyme
# Use system clang or clang without Enzyme plugin
/usr/bin/clang++ -std=c++17 -O3 -fPIC -c \
    -I${SUNDIALS_ROOT}/include \
    -Iinclude \
    src/cvodes_wrapper.cpp \
    -o build/cvodes_wrapper.o

echo "Wrapper object compiled"

# Create static library
ar rcs build/libcvodes_wrapper.a build/cvodes_wrapper.o
echo "Static library created: build/libcvodes_wrapper.a"

echo ""
echo "=== Building dFUSE with CMake ==="
cd build

# Configure CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DDFUSE_BUILD_TESTS=OFF \
    -DDFUSE_BUILD_PYTHON=ON \
    -DDFUSE_USE_ENZYME=ON \
    -DDFUSE_USE_SUNDIALS=ON \
    -DSUNDIALS_ROOT=${SUNDIALS_ROOT}

# Build Python module
make dfuse_core -j

echo ""
echo "=== Build complete ==="
echo "Copy the module: cp build/dfuse_core*.so python/"
