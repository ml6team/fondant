#!/bin/bash
# This script copies the components/ directory to fondant/components, replacing the symlink
# It should be run before building the fondant package'
# This script makes changes to the local files, which should not be committed to git

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

pushd "$parent_path"
rm fondant/components
cp -r components/ fondant/
popd
