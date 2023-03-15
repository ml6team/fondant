# image_caption_component

### Description

This component uses a captioning model ([BLIP](https://github.com/salesforce/BLIP))
to caption the final filtered images for training. 

### **Inputs/Outputs**

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**
* In general, it is [common practice](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) 
to generate relatively short descriptions when finetuning stable diffusion.  
* You can test the captioning of the BLIP model interactively in this
[notebook](../../notebooks/sd-training/dataset_creation.ipynb).
* Captions that are long generally take longer to produce but are more descriptive. However,
it is worth considering first the use case you want to employ.
In general, shorter captions tend to be less prone to errors. As an example, here is an image with generated captions with:  


  * Long captions _(`min_length`=10, `max_length`=30)_: "the image of a santa claus standing with his legs out in the air with his hand on"  

  * Short caption _(`min_length`=20, `max_length`=40)_: "An image of a santa claus cartoon character"  
  
<img alt="png_artifacts" height="200" src="docs/santa_captioned.jpg" width="200" style="display: block; margin: 0 auto"/>
