# Data explorer

## How to use?
You can setup the data explorer container with the `fondant explore` CLI command, which is installed together with the Fondant python package.

```
fondant explore --data-directory LOCAL_FOLDER_TO_MOUNT [--port PORT --container CONTAINER --tag TAG]
```

## Data explorer UI

The data explorer UI enables Fondant users to explore the inputs and outputs of their Fondant pipeline.

The user can load a manifest file, which is produced by Fondant components, which contains metadata about which subsets are available and which columns are present in each subset. After choosing a valid manifest file, a subset (and the columns within the subset) can be explored in 3 tabs.

### Sidebar
In the sidebar, the user can specify the path to a manifest file. This will load the available subsets into a dropdown, from which the user can select one of the subsets. Finally, the columns within the subset are shown in a multiselect box, and can be used to remove / select the columns that are loaded into the exploration tabs.
### Data explorer Tab
The data explorer shows an interactive table of the loaded subset DataFrame with on each row a sample. The table can be used to browse through a partition of the data, to visualize images inside image columns and more.

### Numeric analysis Tab
The numerical analysis tab shows statistics of the numerical columns of the loaded subset (mean, std, percentiles, ...) in a table. In the second part of the tab, the user can choose one of the numerical columns for in depth exploration of the data by visualizing it in a variety of interactive plots.

### Image explorer Tab
The image explorer tab enables the user to choose one of the image columns and analyse these images.