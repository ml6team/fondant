#!/bin/bash
# This script executes the sample pipeline in the example folder, checks the correct execution and
# cleans up the directory again
GIT_HASH=$1

echo "Start integration tests execution ..."

failed_tests=()

# Find all run.sh scripts and execute them
for test_script in ./examples/*/run.sh; do
    test_name=$(basename "$(dirname "$test_script")")

    echo "Running test: $test_name"

    # Set working dir to the currect integration test
    cd $(dirname "$test_script")

    # Execute the run.sh script
    bash ./run.sh $GIT_HASH

    # Check the exit status
    if [ $? -ne 0 ]; then
        echo "Test $test_name failed!"
        failed_tests+=("$test_name")
    fi
done

echo "Tests completed"

if [ ${#failed_tests[@]} -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Failed tests: ${failed_tests[@]}"
    exit 1  # Indicate failure to cicd
fi
