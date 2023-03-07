"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import argparse
import logging

# pylint: disable=import-error
from express.helpers import kfp_helpers

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--project-id',
                    type=str,
                    required=True,
                    help='The id of the gcp-project')


def component(project_id: str) -> None:
    """
    A basic component to check the GPU usage in KFP component
    Args:
        project_id (str): The id of the gcp-project
    """

    logging.info('Started job...')

    logging.info('Project ID: %s', project_id)

    # Show CUDA availability
    kfp_helpers.get_cuda_availability()

    logging.info('Job completed.')


if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    component(ARGS.project_id)
