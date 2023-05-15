#!/bin/bash
# This script copies the components/ directory to fondant/components, replacing the symlink
# It should be run before building the fondant package'
# This script makes changes to the local files, which should not be committed to git
set -e

scripts_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_path=$(dirname "$scripts_path")

pushd "$root_path"
rm fondant/components
cp -r components/ fondant/
popd
