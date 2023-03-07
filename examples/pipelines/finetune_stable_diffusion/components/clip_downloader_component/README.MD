# clip_downloader_component

### Description

This component downloads the images from the list of URLs retrieved from the
[`clip_retireval_components`](../clip_retrieval_component) and
creates the corresponding datasets. It uses the
[img2dataset library](https://github.com/rom1504/img2dataset)
for efficient and fast image download (multi-thread approach). The images are also filtered (based
on size
and area), resized and converted during download.

The components extends the data manifest with two additional namespaces: `knn` and `centroid` to
denote
the datasets of those images.

### **Inputs/Outputs**

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output
parameters.

### **Practical considerations**

* This component can be considered to perform multiple tasks at once -> image downloading,
  filtering,
  formatting and resizing.
* Not all the urls can be downloaded, some urls are broken or some images are filtered out
  depending on size (in practice only around ~75% of the urls are valid).
  At the end of the execution run, a detailed log report to get detailed information on the download
  job execution
  and why some urls failed to download.
* The images are resized to the expected values of stable diffusion (512x512). Since some images are
  not squared,
  the resizing might distort the images. The current resize is set to `border` which resizes the
  images while adding
  a border to the images. You might want to change
  this [parameter](https://github.com/rom1504/img2dataset#:~:text=resize_mode%20The%20way%20to%20resize%20pictures%2C%20can%20be%20no%2C%20border%20or%20keep_ratio%20(default%20border)) 
  depending on your use case.