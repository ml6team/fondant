name: Controlnet aux processor
developer: Bert Christiaens
tags:
  - computer-vision
  - controlnet
  - image-to-image



## Description
This component is based on the `TransformComponent` and uses the controlnet_aux package, which contains lots of useful image processing models, such as Canny edge detection, Openpose, MLSD, depth estimation etc. These image processing models are frequently used for conditional image generation models, such as ControlNet.

## Usage
The component takes an image and calculates a processed image by putting it through a processing model, such as a depth estimator, pose estimator, .. or by applying an algorithm on it, such as Canny edge detection.

Current available processors are ["hed", "midas", "mlsd", "openpose", "pidinet", "normalbae", "lineart", "lineart_coarse", "lineart_anime", "canny", "content_shuffle", "zoe", "mediapipe_face"]

The processor is chosen by passing a processor_id (from the list above) in the Fondant Component.

## Examples
Examples of image cropping by removing the single-color border. Left side is original image, right side is border-cropped image.




| Input image                              | Canny                      | MLSD                      |
|------------------------------------------|----------------------------|---------------------------|
| ![input image](/docs/art/components/controlnet_aux/input.jpg) | ![output image](/docs/art/components/controlnet_aux/output_canny.jpg) | ![output image](/docs/art/components/controlnet_aux/output_mlsd.jpg) | 
| Content shuffle                              | HED                      | Lineart                      |
| ![input image](/docs/art/components/controlnet_aux/output_content_shuffle.jpg) | ![output image](/docs/art/components/controlnet_aux/output_hed.jpg) | ![output image](/docs/art/components/controlnet_aux/output_lineart.jpg) | 
| normalbae                              | openpose                      | pidinet                      |
| ![input image](/docs/art/components/controlnet_aux/output_content_normalbae.jpg) | ![output image](/docs/art/components/controlnet_aux/output_openpose.jpg) | ![output image](/docs/art/components/controlnet_aux/output_pidinet.jpg) | 
