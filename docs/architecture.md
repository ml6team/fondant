# Architecture

### Fondant architecture overview

![fondant architecture](art/architecture.png)

At a high level, Fondant consists of three main parts:

* The `/core` directory serves as the foundational backbone of Fondant, encompassing essential
  shared functionalities:
    * `component_spec.py`: Defines the component spec class which is used to define the
      specification of a component. Those
      specifications mainly include the component image location, arguments, columns it consumes and
      produces.
    * `manifest.py` Describes dataset content, facilitating reference passing between components.
      It evolves during a dataset materialization and aids static evaluation.
    * `schema.py`  Defines the Type class, used for dataset data type definition.
    * `/schema` Directory Containing JSON schema specifications for the component spec and manifest.


* The `/component` directory which contains modules for implementing Fondant components:
    * `component.py`: Defines the `Component` class which is the base class for all Fondant
      components. This is used
      to defines interfaces for different component types (Load, Transform, Write) across different
      data processing frameworks
      (Dask, Pandas, ...). The user should inherit from those classes to implement their own
      components.
    * `data_io.py`: Defines the `DataIO` class which is used to define the reading and writing
      operations from/to a dataset. This includes
      optimizing the reading and writing operations as well as selecting which columns to read/write
      according to the manifest.
    * `executor.py`: Defines the `Executor` class which is used to define the execution of a
      component. This includes
      parsing the component arguments, executing the component and evolving/writing the manifest.
      Each executor
      subclasses a corresponding `Component` class to implement the execution logic for a specific
      component type.


* The `/dataset` directory which contains the modules for implementing a Fondant dataset.
    * `dataset.py`: Defines the `Dataset` class which is used to define the workflow graph to 
      materialize the dataset. The implemented class is then consumed by the compiler to compile  
      to a specific workflow runner.
      This module also implements the `ComponentOp` class which is used to define the component 
      operation in the workflow graph.
    * `compiler.py`: Defines the `Compiler` class which is used to define the compiler that
      compilers the workflow graph for a specific runner.
    * `runner.py`: Defines the `Runner` class which is used to define the runner that executes the
      compiled workflow graph.

### Additional modules

Additional modules in Fondant include:

* `cli.py`: Defines the CLI for interacting with Fondant. This includes the `fondant` command line
  tool which is used to build components,
  compile and run workflows to materialize and explore datasets.
* `explore.py`: Runs the explorer which is a web application that allows the user to explore the
  content of a dataset.
* `build.py`: Defines the `build` command which is used to build and publish a component.
* `testing.py`:  Contains common testing utilities for testing components and datasets.