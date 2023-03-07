# image_classifier_component

### Description

This component implements two different classifiers to filter out
the retrieved images:  

* **_Single component classifier:_** uses classical computer vision algorithm implementation (flood-fill, edge detection)
to ensure that the extracted components are a single component.     

  
* **_Clean-cut classifier_:** uses a pre-trained ML classifier that is trained to identify clean cut images.

### **Inputs/Outputs**

See [`component.yaml`](component.yaml) for a more detailed description on all the input parameters. 

### **Practical considerations**

* The single component classifier is relatively slower than the clean cut classifier. For this reason,
images are first passed through the single component classifier and then (if they are not filtered out) to
the clean cut classifier. 
* The single component classifier is implemented with OpenCV and relies on CPU usage.
* The clean cut classifier is a simple neural networks consisting of a few layers and relies on GPU usage. 


