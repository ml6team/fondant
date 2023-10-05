import typing as t
from pathlib import Path
from glob import glob

import jinja2


def find_components() -> t.List[str]:
    return [Path(d).name for d in sorted(glob("components/*", recursive=True))]


def generate_hub(components) -> str:
    env = jinja2.Environment(
        loader=jinja2.loaders.FileSystemLoader(Path(__file__).parent),
        trim_blocks=True
    )
    template = env.get_template("hub_template.md")

    return template.render(
        components=components
    )


def write_hub(hub: str) -> None:
    with open("docs/components/hub.md", "w") as f:
        f.write(hub)


def main():
    components = find_components()
    hub = generate_hub(components)
    write_hub(hub)


if __name__ == "__main__":
    main()
