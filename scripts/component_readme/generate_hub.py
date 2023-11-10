import typing as t
from pathlib import Path
from glob import glob

import jinja2

from fondant.core.component_spec import ComponentSpec

COMPONENTS_DIR = "components"
COMPONENT_SPEC_FILE = "fondant_component.yaml"
COMPONENT_TYPES = [
    "Data loading",
    "Data writing",
    "Image processing",
    "Data retrieval",
    "Text processing"
]


def read_component_spec(file_path: Path) -> ComponentSpec:
    """Reads and returns the component specification from a file."""
    return ComponentSpec.from_file(file_path / COMPONENT_SPEC_FILE)


def validate_component_type(component_name: str, component_type: t.Optional[str]):
    """Validates that a component type is specified."""
    if component_type is None:
        raise ValueError(f"Component type not specified for `{component_name}`, please "
                         f"add a `component_type` field to the component specification."
                         f"Available types are: {COMPONENT_TYPES}")


def get_components_info() -> t.List[t.Dict[str, str]]:
    """Returns a list of dictionaries containing component information."""
    component_info = []
    for component_file in sorted(glob(f"{COMPONENTS_DIR}/*", recursive=True)):
        component_file = Path(component_file)
        component_spec = read_component_spec(component_file)
        component_name = component_file.name
        component_type = component_spec.component_type

        validate_component_type(component_name, component_type)

        component_info.append({
            'name': component_name,
            'component_type': component_type
        })
    return component_info


def generate_hub(components_info: t.List[t.Dict[str, str]]) -> str:
    """Generates the hub markdown file."""
    env = jinja2.Environment(
        loader=jinja2.loaders.FileSystemLoader(Path(__file__).parent),
        trim_blocks=True
    )
    template = env.get_template("hub_template.md")

    # Group components by component type
    grouped_components = {}
    for component_info in components_info:
        component_type = component_info['component_type']
        if component_type not in grouped_components:
            grouped_components[component_type] = []
        grouped_components[component_type].append(component_info)

    # Sort component types alphabetically
    grouped_components = dict(sorted(grouped_components.items()))

    return template.render(
        components=grouped_components
    )


def write_hub(hub: str) -> None:
    with open("docs/components/hub.md", "w") as f:
        f.write(hub)


def main():
    components_info = get_components_info()
    hub = generate_hub(components_info)
    write_hub(hub)
