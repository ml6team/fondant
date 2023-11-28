"""General configuration for the app."""

SESSION_STATE_VARIABLES = [
    "base_path",
    "pipeline",
    "selected_pipeline_path",
    "run",
    "selected_run_path",
    "component",
    "selected_component_path",
    "partition",
]

DEFAULT_INDEX_NAME = "id"

ROWS_TO_RETURN = 24
