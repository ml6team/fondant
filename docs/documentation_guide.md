# Documentation Guide

## Getting started with Fondant

Learn about the Fondant project and how to get started with it.

→ Start with the official guide on how to [install](guides/installation.md) Fondant.  
→ Get started by running your first fondant [dataset](guides/first_dataset.md) using the [local
runner](runners/local.md).  
→ Learn how to build your own [Fondant Dataset](guides/build_a_simple_dataset.md) and implement your 
own [custom components](guides/implement_custom_components.md).    
→ Learn how to use the [data explorer](data_explorer.md) to explore your dataset.

## Fondant fundamentals

Learn how to use Fondant to build your own datasets.

-> Build your own fondant [dataset](dataset.md) using the Fondant dataset SDK.  
-> Use existing [reusable components](components/hub.md) to build your dataset.  
-> Build your own custom [lightweight component](components/lightweight_components.md) 
and share them by packaging them into [containerized component](components/containerized_components.md) using the Fondant component
SDK.  
-> Learn how to publish your own [components](components/publishing_components.md) to a container
registry so that you can reuse them in your workflows.  

## Components hub

Have a look at the [components hub](components/hub.md) to see what components are available.

## Fondant Runners

Learn how to run your Fondant workflow to materialize your datasets on different platforms.

<table class="images" width="100%" style="border: 0px solid white; width: 100%; text-align: center;">
    <tr style="border: 0px;">
        <td width="25%" style="border: 0px; width: 25%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/docker_compose.png?raw=true"  style="height: 150px; margin-left: auto; margin-right: auto;" />
                <figcaption class="caption"><strong>LocalRunner</strong></figcaption>
            </figure>
        </td>
        <td width="25%" style="border: 0px; width: 25%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/vertex_ai.png?raw=true"  style="height: 150px; margin-left: auto; margin-right: auto;" />
                <figcaption class="caption"><strong>VertexRunner</strong></figcaption>
            </figure>
        </td>
        <td width="25%" style="border: 0px; width: 25%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/kubeflow_pipelines.png?raw=true"  style="height: 150px; margin-left: auto; margin-right: auto;" />
                <figcaption class="caption"><strong>KubeflowRunner</strong></figcaption>
            </figure>
        </td>
        <td width="25%" style="border: 0px; width: 25%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/sagemaker.png?raw=true"  style="height: 150px; margin-left: auto; margin-right: auto;" />
                <figcaption class="caption"><strong>SageMakerRunner</strong></figcaption>
            </figure>
        </td>
    </tr>
</table>



-> [LocalRunner](runners/local.md): ideal for developing fondant datasets and components faster.   
-> [VertexRunner](runners/vertex.md): used for running a fondant workflows on Vertex AI.  
-> [KubeflowRunner](runners/kfp.md): used for running a fondant workflows on a Kubeflow cluster.  
-> [SageMakerRunner](runners/sagemaker.md): used for running a fondant workflows on a SageMaker.

## Fondant Explorer

Discover how to utilize the Fondant [data explorer](data_explorer.md) to investigate your dataset, including visualizing intermediary steps between components.

## Advanced Concepts

Learn about some of the more advanced concepts in Fondant.

-> Learn more about the [architecture](architecture.md) of Fondant and how it works under the
hood.  
-> Learn how Fondant uses [caching](caching.md) to speed up your dataset development.  
-> Find out how Fondant uses [partitions](partitions.md) to parallelize and scale your workflows and
how you can use it to your advantage.  
-> Learn how to setup a Kubeflow to run your Fondant workflow on a [Kubeflow cluster](runners/kfp_infrastructure.md).

## Contributing

Learn how to contribute to the Fondant project through
our [contribution guidelines](contributing.md).

## Announcements

Check out our latest [announcements](blog/index.md) about Fondant.
