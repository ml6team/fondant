"""
This is a dummy valid loading component.
"""
import logging

from fondant.component import FondantLoadComponent
from fondant.component_spec import FondantComponentSpec
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class LoadFromHubComponent(FondantLoadComponent):
    def __init__(self):
        self.spec = FondantComponentSpec.from_file("tests/example_specs/valid_component/fondant_component.yaml")
        self.args = self._add_and_parse_args()
    
    def load(self):
        return -1


if __name__ == "__main__":
    component = LoadFromHubComponent()
    component.run()