"""
Pylint script to run pylint from the root of a module. This ensures that
all paths and imports are resolved correctly.
"""

import os
import sys
import logging
import argparse
import subprocess  # nosec
from typing import Any
from pathlib import Path
from setuptools import find_packages

# pylint: disable=invalid-name
parser = argparse.ArgumentParser()
module_parser = parser.add_mutually_exclusive_group()
module_parser.add_argument('--module',
                           help='Module to pylint',
                           type=str,
                           action='append')
module_parser.add_argument('--modules-file',
                           help='Path to file with modules',
                           type=str)

GREEN, RED, NC = '\033[0;32m', '\033[0;31m', '\033[0m'


def install_requirements(module_path: str) -> None:
    """Install the requirements.txt file of a module on best effort basis. If
    the installation fails, the error is printed but ignored.

    Args:
        module_path: Path to the root of the module.

    """
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',  # nosec
                        'requirements.txt', '--force-reinstall'],
                       cwd=module_path,
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error('Failed to install requirements')
        logging.error(e.stderr.decode())


def pylint_module(module_path: str) -> int:
    """Run pylint on a module with correct path configuration.

    Args:
        module_path: Path to the module.

    Returns:
        Pylint return code.

    """
    print(f'Linting module {module_path}')
    requirements = Path(os.path.join(module_path, "requirements.txt"))
    if requirements.exists():
        install_requirements(module_path)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    init_hook = f'import sys; sys.path.append(r"{module_path}")'

    process = subprocess.run(['pylint', module_path,  # nosec
                              f'--rcfile={os.path.join(dir_path, ".pylintrc")}',
                              f'--init-hook={init_hook}'],
                             check=False)

    return process.returncode


def main(args: Any) -> None:
    """Run pylint on provided modules."""
    if args.modules_file:
        # Read in all modules to be pylinted
        with open(args.modules_file, 'r') as f:
            modules = [line.strip('\n') for line in f.readlines()]
    elif args.module:
        modules = args.module
    else:
        # If no modules were given, detect them automatically
        logging.warning('No modules were supplied, detecting them automatically')
        modules = [module for module in find_packages() if '.' not in module]

    logging.warning('Running pylint for %s', modules)

    # Run pylint for each module from the module directory
    failures = [module for module in modules
                if pylint_module(os.path.abspath(module)) != 0]

    # Print some encouraging messages with info
    if failures:
        print(RED + "Failure :'(" + NC)
        print("Following modules failed: {}".format(failures))
        sys.exit(1)
    else:
        print(GREEN + 'Success!' + NC)


if __name__ == '__main__':
    main(parser.parse_args())
