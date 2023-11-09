# Documentation Guide

## Getting started with Fondant

Learn about the Fondant project and how to get started with it.

→ Start with the official guide on how to [install](guides/installation.md) Fondant.  
→ Get started by running your first fondant [pipeline](guides/first_pipeline.md) using the [local
runner](runners/local.md). 
→ Learn how to build your own [Fondant Pipeline](guides/build_a_simple_pipeline.md) and implement your 
own [custom components](guides/implement_custom_components.md).  
→ Learn how to use the [data explorer](data_explorer.md) to explore the outputs of your pipeline.

## Fondant fundamentals

Learn how to use Fondant to build your own data processing pipeline.

-> Design your own fondant [pipeline](pipeline.md) using the Fondant pipeline API.  
-> Use existing [reusable components](components/hub.md) to build your pipeline.  
-> Use [generic components](components/generic_component.md) to load/write your custom data format
to/from Fondant.  
-> Build your own [custom component](components/custom_component.md) using the Fondant component
API.  
-> Learn how to publish your own [components](components/publishing_components.md) to a container
registry so that you can reuse them in your pipelines.

## Components hub

Have a look at the [components hub](components/hub.md) to see what components are available.

## Fondant Runners

Learn how to run your Fondant pipeline on different platforms.

<table class="images" width="100%" style="border: 0px solid white; width: 100%;">
    <tr style="border: 0px;">
        <td width="25%" style="border: 0px; width: 28.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/docker_compose.png?raw=true" />
                <figcaption class="caption"><strong>LocalRunner</strong></figcaption>
            </figure>
        </td>
        <td width="25%" style="border: 0px; width: 30.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/vertex_ai.png?raw=true" />
                <figcaption class="caption"><strong>VertexRunner</strong></figcaption>
            </figure>
        </td>
        <td width="25%" style="border: 0px; width: 30.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/kubeflow_pipelines.png?raw=true" />
                <figcaption class="caption"><strong>KubeflowRunner</strong></figcaption>
            </figure>
        <td width="25%" style="border: 0px; width: 33.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/sagemaker.png?raw=true" />
                <figcaption class="caption"><strong>🚧SageMakerRunner🚧</strong></figcaption>
            </figure>
        </td>
    </tr>
</table>

<style>
    .caption {
        text-align: center; /* Adjust the alignment as needed */
    }
</style>

-> [LocalRunner](runners/local.md): ideal for developing fondant pipelines and components faster.   
-> [VertexRunner](runners/vertex.md): used for running a fondant pipeline on Vertex AI.  
-> [KubeflowRunner](runners/kfp.md): used for running a fondant pipeline on a Kubeflow cluster.  
-> [SageMakerRunner](runners/kfp.md): used for running a fondant pipeline on a SageMaker pipelines (
🚧 Coming Soon 🚧).

## Fondant Explorer

Discover how to utilize the Fondant [data explorer](data_explorer.md) to navigate your pipeline
outputs, including visualizing intermediary steps between components.

## Advanced Concepts

Learn about some of the more advanced concepts in Fondant.

-> Learn more about the [architecture](architecture.md) of Fondant and how it works under the
hood.  
-> Understand how Fondant passes data between components with the [manifest](manifest.md).  
-> Learn how Fondant uses [caching](caching.md) to speed up your pipeline development.  
-> Find out how Fondant uses [partitions](partitions.md) to parallelize and scale your pipeline and
how you can use it to your advantage.

## Contributing

Learn how to contribute to the Fondant project through
our [contribution guidelines](contributing.md).

## FAQ

Browse through the [frequently asked questions](faq.md) about Fondant.

## Announcements

Check out our latest [announcements] about Fondant.

-> 25 million Creative Commons image dataset released. Read more about it [here](announcements/CC_25M_press_release.md).

