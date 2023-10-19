import argparse
from pathlib import Path

import jinja2
from fondant.component_spec import ComponentSpec


def read_component_spec(component_spec_path: Path) -> ComponentSpec:
    return ComponentSpec.from_file(component_spec_path)


def generate_readme(component_spec: ComponentSpec, *, component_dir: Path) -> str:
    env = jinja2.Environment(
        loader=jinja2.loaders.FileSystemLoader(Path(__file__).parent),
        trim_blocks=True
    )
    env.filters["eval"] = eval

    template = env.get_template("readme_template.md")

    return template.render(
        id=component_dir.name,
        name=component_spec.name,
        description=component_spec.description,
        consumes=component_spec.consumes,
        produces=component_spec.produces,
        arguments=[arg for arg in component_spec.args.values()
                   if arg.name not in component_spec.default_arguments],
        tests=(component_dir / "tests").exists()
    )


def write_readme(readme: str, component_dir: Path) -> None:
    with open(component_dir / "README.md", "w") as f:
        f.write(readme)


def main(component_spec_path: Path):
    component_spec = read_component_spec(component_spec_path)
    component_dir = component_spec_path.parent
    readme = generate_readme(component_spec, component_dir=component_dir)
    write_readme(readme, component_dir=component_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component_specs",
                        nargs="+",
                        type=Path,
                        help="Path to the component spec to generate a readme from")
    args = parser.parse_args()

    for spec in args.component_specs:
        main(spec)
