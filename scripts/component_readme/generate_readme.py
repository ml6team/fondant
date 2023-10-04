import argparse
from pathlib import Path

import jinja2
from fondant.component_spec import ComponentSpec


def read_component_spec(component_dir: Path) -> ComponentSpec:
    return ComponentSpec.from_file(component_dir / "fondant_component.yaml")


def generate_readme(component_spec: ComponentSpec, *, id_: str) -> str:
    template_path = Path(__file__).with_name("readme_template.md")
    with open(template_path, "r") as f:
        template = jinja2.Template(f.read(), trim_blocks=True)

    return template.render(
        id=id_,
        name=component_spec.name,
        description=component_spec.description,
        consumes=component_spec.consumes,
        produces=component_spec.produces,
        arguments=component_spec.args.values(),
    )


def write_readme(readme: str, component_dir: Path) -> None:
    with open(component_dir / "README.md", "w") as f:
        f.write(readme)


def main(component_dir: Path):
    component_spec = read_component_spec(component_dir)
    readme = generate_readme(component_spec, id_=component_dir.name)
    write_readme(readme, component_dir=component_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--component_dir",
                        type=Path,
                        help="Path to the component to generate a readme for")
    args = parser.parse_args()

    main(args.component_dir)
