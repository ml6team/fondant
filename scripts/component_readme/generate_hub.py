import typing as t
from pathlib import Path
from glob import glob

import jinja2

from fondant.core.component_spec import ComponentSpec

HUB_FILE = "docs/components/hub.md"
HUB_TEMPLATE_FILE = "hub_template.md"
COMPONENTS_DIR = "components"
COMPONENT_SPEC_FILE = "fondant_component.yaml"
COMPONENT_TYPE_TAGS = [
    "Data loading",
    "Data writing",
    "Data retrieval",
    "Image processing",
    "Text processing",
    "Multi-modal processing",
    "Audio processing",
    "Video processing",
]


def read_component_spec(file_path: Path) -> ComponentSpec:
    """Reads and returns the component specification from a file."""
    return ComponentSpec.from_file(file_path / COMPONENT_SPEC_FILE)


def validate_component_type_tags(component_name: str, tags: t.Optional[t.List[str]]):
    """Validates that a valid component type is specified."""
    if tags is None:
        raise ValueError(f"Component type not specified for `{component_name}`, please "
                         f"add a `tags` field to the component specification with a "
                         f"corresponding component type tag. \n"
                         f"Available component tags are: {COMPONENT_TYPE_TAGS}")

    if not any(value in tags for value in COMPONENT_TYPE_TAGS):
        raise ValueError(
            f"Component type not found for `{component_name}`, specified tags are: {tags}. \n"
            f"Make sure that a component"
            f" type tag is specified in the `tags` field of the component specification. \n"
            f"Available component tags are: {COMPONENT_TYPE_TAGS}")


def get_components_info() -> t.List[t.Dict[str, t.Any]]:
    """Returns a list of dictionaries containing component information."""
    component_info = []
    for component_file in sorted(glob(f"{COMPONENTS_DIR}/*", recursive=True)):
        component_file = Path(component_file)
        component_spec = read_component_spec(component_file)
        component_dir = component_file.name
        component_name = component_spec.name
        tags = component_spec.tags

        validate_component_type_tags(component_name, tags)

        component_info.append({
            'dir': component_dir,
            'name': component_name,
            'tags': tags
        })
    return component_info


def group_and_sort_components(components_info: t.List[t.Dict[str, t.Any]]) -> t.Dict[str, str]:
    """Groups components by component tag and sorts them alphabetically."""
    grouped_components = {}
    for component_info in components_info:
        for tag in component_info['tags']:
            grouped_components.setdefault(tag, []).append({
                'dir': component_info['dir'],
                'name': component_info['name'],
                'tag': tag,
            })

    return dict(sorted(grouped_components.items()))


def generate_hub(components_info: t.List[t.Dict[str, str]]) -> str:
    """Generates the hub markdown file."""
    env = jinja2.Environment(
        loader=jinja2.loaders.FileSystemLoader(Path(__file__).parent),
        trim_blocks=True
    )
    template = env.get_template(HUB_TEMPLATE_FILE)

    grouped_components = group_and_sort_components(components_info)

    return template.render(
        components=grouped_components
    )


def write_hub(hub: str) -> None:
    with open(HUB_FILE, "w") as f:
        f.write(hub)


def main():
    components_info = get_components_info()
    hub = generate_hub(components_info)
    write_hub(hub)


if __name__ == "__main__":
    main()
