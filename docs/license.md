# License

Fondant is distributed under the [OpenRAIL-S license](https://www.licenses.ai/source-code-license), 
a permissive license that allows for commercial and non-commercial usage.

RAIL stands for `Responsible AI Licenses`, which empower developers to restrict the use of their AI 
technology in order to prevent irresponsible and harmful applications. These licenses include 
behavioral-use clauses which restrict certain use-cases.

The `Open-` prefix indicates that it permits free use and distribution, albeit subject to use 
restrictions. It aims to clarify, on its face, that the licensor offers royalty free access and 
flexible downstream use and re-distribution of the licensed material or any derivatives of it.

The `-S` suffix indicates that the license applies to source code. Alternative license versions 
are available to license data, applications, or models.

## Adoption

Early adopters of the OpenRAIL licenses were:
- [Hugging Face](https://huggingface.co/blog/open_rail)
- [BigScience](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)
- [stability.ai](https://stability.ai/blog/stable-diffusion-public-release#:~:text=The%20model,service%20on%20it.) 

Adoption of the OpenRAIL licenses had already 
[increased to 9.81% of all model repositories](https://openfuture.pubpub.org/pub/growth-of-responsible-ai-licensing/release/2) 
on the Hugging Face Hub by 24 January 2023.

## FAQ

#### What are we licensing?
The license covers the Fondant framework and provided reusable components.

#### Is this an open source license?
This is not an open source license according to the Open Source Initiative definition, because it 
has some restrictions on the use of the model. That said, it does not impose any restrictions on 
reuse, distribution, commercialization, adaptation as long as the software is not being applied 
towards use-cases that have been restricted.

#### Does the license allow commercial use?
Yes, as long as the software is not applied towards use-cases that have been restricted.

#### How does the license compare with other open source licenses?
The license is closest to the Apache 2.0 open source license but it includes specific conditions 
under which restrictions to the USE, REPRODUCTION, AND DISTRIBUTION of source code can apply.   

#### Do the use case restrictions apply to datasets generated using Fondant?
Yes. As the license prohibits Fondant to be used towards any of the restricted use cases, it cannot 
be used to generate datasets for those use cases. Generated datasets should integrate the same 
use-based restrictions as part of their license.
