#!/bin/bash
# This script reverts the src/fondant/components directory to a symlink to the components/
# directory.
# It should be run after running scripts/pre-build.sh and building the fondant package
set -e

scripts_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_path=$(dirname "$scripts_path")

pushd "$root_path"
rm -rf src/fondant/components
pushd src/fondant
ln -s ../../components
popd
popd
