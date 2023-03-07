# sd_finetuning_component

### Description

This component takes as input the
final data manifest that is output by the dataset
creation pipeline. The data
manifest keeps reference of all the necessary metadata is required for the next step such as the
reference to the filtered images and captions. The component prepares the dataset for training
according to the
required [format](https://huggingface.co/docs/datasets/image_dataset#:~:text=in%20load_dataset.-,Image%20captioning,-Image%20captioning%20datasets)
and starts the finetuning jobs.
### **Inputs/Outputs**

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**
* The script uses the official [finetuning script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
provided by diffusers and is launched with a subprocess call. Check the following
[docs](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) for more information on finetuning.
* Finetuning stable diffusion models requires at least 30 GB of GPU memory. It is currently recommended to
use at least 2 A100 GPUs for finetuning. Make sure to configure `accelerate_config.yaml` if you want to change
the number of GPUs used for training. 
* It is difficult to pinpoint exactly how many steps are needed to obtain a good finetunable model. The best approach
is to create checkpoints throughout training and visualize the progression of the generation with a fixed prompt and seed.
* A good starting point is to finetune for 2 epochs (2 full passes over the dataset).