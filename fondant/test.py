import yaml

from fondant.component_spec import FondantComponentSpec
from fondant.manifest import Manifest


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            spec = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return spec


load_from_hub_spec = FondantComponentSpec(
    specification=read_yaml(
        "/Users/nielsrogge/Documents/python_projects/express/examples/pipelines/simple_pipeline/components/load_from_hub/src/fondant_component.yaml"
    )
)

image_filtering_spec = FondantComponentSpec(
    specification=read_yaml(
        "/Users/nielsrogge/Documents/python_projects/express/examples/pipelines/simple_pipeline/components/image_filtering/src/fondant_component.yaml"
    )
)

# 1. create new manifest
manifest = Manifest.create(base_path=".", run_id="100", component_id="1")
print("Initial manifest:", manifest)

# 2. evolve manifest based on load_from_hub component spec
component_id = "2"
output_manifest = manifest.evolve(component_spec=load_from_hub_spec)

print("Intermediate manifest:", output_manifest)

# 3. evolve manifest based on image_filtering component spec
component_id = "3"
output_manifest = output_manifest.evolve(component_spec=image_filtering_spec)

print("Output manifest:", output_manifest)
