## Building and publishing components

Before you can use your custom component within your pipeline you have to build the component into a
Docker container. The Fondant CLI offers a specialized `fondant build` command designed
explicitly for this task.

=== "Console"

    ```bash
    fondant build <component dir> -t <image tag>
    ```

    It's important to note that the `fondant build` command offers additional arguments. To access a
    complete list of all available arguments, you can execute `fondant build -h`.

=== "Python"

    ```python
    from fondant.build import build_component

    component_dir = <path_to_component_dir>
    tag = <optiona_docker_tag>
    build_component(
        component_dir=component_dir,
        tag=tag,
    )
    ```

Ensure that the component directory refers to the accurate path of the directory where your
component is located.

The tag arguments is used to specify the Docker container tag. When specified, the tag in the
referenced component specification yaml will also be
updated, ensuring that the next pipeline run correctly references the image.


!!! note "IMPORTANT"   

    When developing custom components using the local runner, building and publishing components is not required.
    That is because components that are not located in the registry (local custom components) will be built during runtime when using the local runner. 
    This allows for quicker iteration during component development.


