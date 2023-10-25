# Data explorer

## How to use?
You can setup the data explorer container with the `fondant explore` CLI command, which is installed together with the Fondant python package.

```
fondant explore [--base_path BASE_PATH] [--container CONTAINER] [--tag TAG] [--port PORT] [--credentials CREDENTIALS]
```

Where the base path can be either a local or remote base path. Make sure to pass the proper mount credentials arguments when using a remote base path or a local base path 
that references remote datasets. You can do that either with `--auth-gcp`, `--auth-aws` or `--auth-azure` to
mount your default local cloud credentials to the pipeline. Or You can also use the `--credentials` argument to mount custom credentials to the local container pipeline.

Example: 

```bash
fondant explore --base_path gs://foo/bar --auth-gcp
```
## Data explorer UI

The data explorer UI enables Fondant users to explore the inputs and outputs of their Fondant pipeline.

The user can specify a pipeline and a specific pipeline run and component to explore. The user will then be able to explore the different subsets produced by by Fondant components.

The chosen subset (and the columns within the subset) can be explored in 3 tabs.

### Sidebar
In the sidebar, the user can specify the path to a manifest file. This will load the available subsets into a dropdown, from which the user can select one of the subsets. Finally, the columns within the subset are shown in a multiselect box, and can be used to remove / select the columns that are loaded into the exploration tabs.
### Data explorer Tab
The data explorer shows an interactive table of the loaded subset DataFrame with on each row a sample. The table can be used to browse through a partition of the data, to visualize images inside image columns and more.

### Numeric analysis Tab
The numerical analysis tab shows statistics of the numerical columns of the loaded subset (mean, std, percentiles, ...) in a table. In the second part of the tab, the user can choose one of the numerical columns for in depth exploration of the data by visualizing it in a variety of interactive plots.

### Image explorer Tab
The image explorer tab enables the user to choose one of the image columns and analyse these images.