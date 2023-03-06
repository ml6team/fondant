#!/bin/bash

# This script provides a wrapper to easily run the license checker.
# You can either provide a single module using -m, or a file with a list of modules using -mf
# If nothing is provided, the license checker runs on the working directory

# Saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

# Argument parsing
while [[ "$#" -gt 0 ]]; do case $1 in
  -m|--module) module="$2"; shift;;
  -mf|--modules-file) modules_file="$2"; shift;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

MODULES=()
# Build the list of modules to check from the input arguments
if [ -n "${module-}" ] && [ -n "${modules_file-}" ]; then
    echo "Both a module and modules_file were provided. Please only provide one."
    exit 1
elif [ -n "${module-}" ]; then
    MODULES+=("${module}")
elif [ -n "${modules_file-}" ]; then
    # List all modules in pylint-modules
    dir_name="$(dirname "${modules_file}")"
    while IFS= read -r line || [[ "$line" ]]; do
      MODULES+=("${dir_name}/${line}");
    done < "${modules_file}"
else
    MODULES+=(".")
fi

echo "Running license check for modules"
echo

# Install pip-tools for pip-compile but suppress stdout
pip install pip-tools > /dev/null

if [ ${#MODULES[@]} -eq 0 ]
then
    echo "No modules found."
else
    for module in "${MODULES[@]}"; do
        echo "Checking ${module}"

        # Try setup.py if it exists, otherwise requirements.txt
        if [ -e "${module%/}"/setup.py ]
        then
            file="${module%/}"/setup.py
        else
            file="${module%/}"/requirements.txt
        fi
        echo "Using ${file}"

        pip-compile "${file}" -q -o /tmp/requirements.txt
        echo "Installing dependencies..."
        pip install -q -r /tmp/requirements.txt
        liccheck --no-deps -s license_strategy.ini -r /tmp/requirements.txt -l paranoid
    done
fi
