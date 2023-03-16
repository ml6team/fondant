# Pipelines for fine-tuning Stable Diffusion

This folder contains 2 separate KubeFlow pipelines:

* [Dataset Creation Pipeline](#1-dataset-creation-pipeline): creates the dataset that will be used
  to finetune the Stable Diffusion model
* [Stable Diffusion Finetuning pipeline](#2-sd-finetuning-pipeline): finetunes a pretrained Stable
  Diffusion model on the created dataset.

The separation between the two pipelines enables to run them disjointly to allow for quick
and fast experimentation; the dataset creation pipeline will in practice only be run once in a while to collect large amount of images. The Stable Diffusion finetuning pipeline on the other hand might be run several times to allow experimentation
with different hyperparameters or to resume training from a certain checkpoint.

## [1. Dataset Creation pipeline](pipelines/dataset_creation_pipeline.py)

### Description

The aim of this pipeline is to prepare a dataset to finetune a pre-trained Stable Diffusion on, based on a set of seed images.

In short, the pipeline first loads in a reference (seed) dataset containing curated images of a certain style/domain
(e.g. clip art images) and then retrieves similar images to this dataset from the [LAION dataset](https://laion.ai/), a large scale public dataset containing around 5 billion images. The retrieved images then undergo additional filtering
steps to ensure that they are of good quality (e.g. single component, clean-cut). Finally, the remaining images
are then captioned using a captioning model to generate image/caption pairs for Stable Diffusion finetuning.

### Pipeline steps

The pipeline consists of the following components:

**[1) Dataset loader component:](components/dataset_loader_component)** This component
loads in an image dataset from a specific [Google Cloud Storage](https://cloud.google.com/storage/docs) path and creates a
parquet file with relevant metadata (image path, format, size...).

**[2) Image filter component:](components/image_filter_component)** This component is
used to filter the images based on the metadata
attributes to only keep images of a certain size and format.

**[3) Image conversion component:](components/image_conversion_component)** This
component converts images from different selected formats
(currently only `png` and `svg`) to `jpg` images. This step is necessary since other images formats
are not suitable for training ML models on and often contain artifacts that can be eliminated
during conversion.

**[4) Image embedding component:](components/image_embedding_component)** This component
extracts [image embeddings](https://rom1504.medium.com/image-embeddings-ed1b194d113e)
from the converted images using
a [CLIP model](https://www.google.com/search?q=clip+embeddings&oq=clip+embeddings&aqs=chrome..69i57j0i22i30j69i60j69i64l2j69i60j69i64j69i60.6764j0j7&sourceid=chrome&ie=UTF-8).  
Since image embeddings are good at capturing the features of the image in a compact and useful way,
it
will be used in the next steps to retrieve images similar to our seed images.

[**5) Clip retrieval component:**](components/clip_retrieval_component) This component
retrieves images from the LAION dataset using a clip
retrieval system. The images are retrieved using an efficient index built from the previously
extracted embeddings that enables for fast and accurate
querying against a large database. Checkout
this [link](https://github.com/rom1504/clip-retrieval) for information on clip retrieval. You can
also
test out clip retrieval in
this [demo](https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn5.laion.ai&index=laion5B&useMclip=false).
The output of this component is a list of URLs containing the link to the retrieved image.

[**6) Clip downloader component:**](components/clip_downloader_component) This component
downloads the images from the list of URLs and
creates the corresponding datasets. It uses the
following [library](https://github.com/rom1504/img2dataset)
for efficient and fast image download (multi-thread approach). The images are also filtered (based
on size
and area), resized and converted during download.

[**7) Image caption component:**](components/image_caption_component) This component
uses a captioning
model ([BLIP](https://github.com/salesforce/BLIP))
to caption the final filtered images for training.

## [2. SD finetuning pipeline](pipelines/sd_finetuning_pipeline.py)

### Description

This pipeline aims to finetune a pre-trained stable diffusion model on the collected dataset.

### Pipeline steps

This pipeline consists only of a single component:

**[1) SD finetuning component:](components/sd_finetuning_component)** takes as input the
final data manifest that is output by the dataset
creation pipeline. The data
manifest keeps reference of all the necessary metadata is required for the next step such as the
reference to the filtered images and captions. The component prepares the dataset for training
according to the
required [format](https://huggingface.co/docs/datasets/image_dataset#:~:text=in%20load_dataset.-,Image%20captioning,-Image%20captioning%20datasets)
and starts the finetuning jobs.

# Building the images

To build and push the component docker images to the container registry, execute the following command:

```bash
bash build_images.sh
```

This will build all the components located in the `components` folder, you could also opt for building a specific component
by passing the `--build-dir` and passing the folder name of the component you want to build. 


## **Data Manifest: a common approach to simplify different steps throughout the pipeline**
In order to keep track of the different data sources, we opt for a manifest-centered approach where 
a manifest is simply a JSON file that is passed and modified throughout the different steps of the pipeline. 

```json
{
   "dataset_id":"<run_id>-<component_name>",
   "index":"<path to the index parquet file>",
   "associated_data":{
      "dataset":{
         "namespace_1":"<path to the dataset (metadata) parquet file of the datasets associated with `namespace_1`>",
         "...":""
      },
      "caption":{
         "namespace_1":"<path to the caption parquet file associated with `namespace_1`>",
         "...":""
      },
      "embedding":{
         "namespace_1":"<remote path to the directory containing the embeddings associated with `namespace_1`",
         "...":""
      }
   },
   "metadata":{
      "branch":"<the name of the branch associated with the component>",
      "commit_hash":"<the commit of the component>",
      "creation_date":"<the creation date of the manifest>",
      "run_id":"<a unique identifier associated with the kfp pipeline run>"
   }
}
```
Further deep dive on some notations:  

* **namespace:** the namespace is used to identify the different data sources. For example, you can give 
your seed images a specific namespace (e.g. `seed`). Then, the images retrieved with clip-retrieval will 
have different namespace (e.g. `knn`, `centroid`).

* **index**: the index denotes a unique index for each images with the format <namespace_uid> (e.g. `seed_00010`).
It indexes all the data sources in `associated_data`.
**Note**: the index keeps track of all the namespace (e.g. [`seed_00010`,`centroid_0001`, ...])

* **dataset**: a set of parquet files for each namespace that contain relevant metadata
(image size, location, ...) as well as the index.

* **caption**: a set of parquet files for each namespace that contain captions
image captions as well as the index.

* **metadata**: Helps keep track of the step that generated that manifest, code version and pipeline run id.

The Express pipeline consists of multiple steps defines as **Express steps** that are repeated 
throughout the pipeline. The manifest pattern offers the required flexibility to promote its reuse and avoid
duplication of data sources. For example:  

* **Data filtering** (e.g. filtering on image size): add new indices to the `index` but retain associated data.  

* **Data creation** (e.g. clip retrieval): add new indicies to the new `index` and another source of data under associated data with a new namespace.  

* **Data transformation** (e.g. image formatting): retain indices but replace dataset source in `dataset`.  
