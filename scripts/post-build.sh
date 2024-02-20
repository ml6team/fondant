#!/bin/bash
# This script reverts the `scripts/pre-build.sh` script by moving the package code back into a
# `src` directory.
# It should be run after running scripts/pre-build.sh and building the fondant package
set -e

scripts_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_path=$(dirname "$scripts_path")

pushd "$root_path"
mkdir -p src/fondant
mv fondant/* src/fondant
rm -rf fondant
popd
