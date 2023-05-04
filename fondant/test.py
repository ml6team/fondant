from fondant.component_spec import FondantComponentSpec
import yaml

with open(
    "/Users/nielsrogge/Documents/python_projects/express/examples/pipelines/simple_pipeline/components/image_filtering/src/fondant_component.yaml"
) as c:
    specification = yaml.safe_load(c)

spec = FondantComponentSpec(specification)

print(spec)

print([arg.name for arg in spec.args])
