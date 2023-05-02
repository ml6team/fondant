import yaml

from fondant.component_spec import FondantComponentSpec
from fondant.manifest import Manifest

with open(
    "/Users/nielsrogge/Documents/python_projects/express/examples/pipelines/simple_pipeline/components/load_from_hub/src/fondant_component.yaml",
    "r",
) as stream:
    try:
        spec = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

component_spec = FondantComponentSpec(specification=spec)

print("Component spec:", component_spec)

# create new manifest
manifest = Manifest.create(base_path=".", run_id="100", component_id="200")
print("Initial manifest:", manifest)

# evolve manifest based on component spec
output_manifest = manifest.evolve(component_spec=component_spec)

print("Output manifest:", output_manifest)
