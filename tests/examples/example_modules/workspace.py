from fondant.dataset import Workspace


def create_workspace_with_args(name):
    return Workspace(name=name, base_path="some/path")


def create_workspace():
    return Workspace(name="test_workspace", base_path="some/path")


def not_implemented():
    raise NotImplementedError


workspace = create_workspace()


number = 1
