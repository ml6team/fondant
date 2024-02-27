#!/bin/bash
# This script unpacks the src folder before building to make the poetry `include` and `exclude`
# sections work correctly for both sdist and wheel
# (https://github.com/python-poetry/poetry/issues/8994)
# Building without running this script also works, but includes the full code for the components.
# This script makes changes to the local files, which should not be committed to git. Run
# scripts/post-build.sh to clean them up.
set -e

scripts_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_path=$(dirname "$scripts_path")

pushd "$root_path"
mkdir fondant
mv src/fondant/* fondant
rm -rf src
popd
