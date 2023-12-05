### SageMaker Runner

Leverage [AWS SageMaker](https://aws.amazon.com/sagemaker/) to run your Fondant pipelines.

This makes it easy to scale up your pipelines in a serverless manner without worrying about infrastructure
deployment.

The Fondant SageMaker runner will compile your pipeline to a SageMaker pipeline spec and submit it to SageMaker.


### Installing the SageMaker runner

Make sure to install Fondant with the SageMaker runner extra.

```bash
pip install fondant[sagemaker]
```

### Prerequisites

- You will need to have an AWS account and have the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) installed and configured.
- Fondant on SageMaker uses an s3 bucket to store the pipeline artifacts. You will need to create an s3 bucket and make sure that the role you are using has access to it. You can create a bucket using the AWS CLI:
    ```bash
    aws s3 create-bucket --bucket <bucket_name> --region <region>
    ```
You can then set this bucket as the `base_path` of your pipeline with the syntax: `s3://<bucket_name>/<path>`.
-  Setup a [SageMaker domain](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)
TODO: try with new domains on fresh account

### Running a pipeline with SageMaker


Since compiling a sagemaker spec requires access to the AWS SageMaker API, you will need to be logged in to 
AWS with a role that has all the required permissions:
- TODO: add instructions




=== "Console"
    
    ```bash 
    fondant run sagemaker <pipeline_ref> \
     --role-arn $SAGEMAKER_ROLE_ARN 
    ```
    
    The pipeline ref is reference to a fondant pipeline (e.g. `pipeline.py`) where a pipeline instance
    exists.


=== "Python"
    
    ```python
    from fondant.pipeline.runner import SageMakerRunner
    
    runner = SageMakerRunner()
    runner.run(
        input=<path_to_pipeline>,
        role_arn=<role_arn>,
        pipeline_name=<pipeline_name>
        instance_type=<instance_type>,
    )
    ```


Once your pipeline is running you can monitor it using the SageMaker [Studio](https://aws.amazon.com/sagemaker/studio/).

!!! note "IMPORTANT"

    If the Fondant SageMaker runner takes a Fondant pipeline object as input it will compile it first and then run it subsequently. You can also use the SageMaker compiler to compile your pipeline and then run it with the SageMaker runner by specifying the compiled spec as input to the runner.



#### Using custom Fondant components on SageMaker

SageMaker only supports images hosted on a private ECR registry. If you want to use custom Fondant components on SageMaker you will need to build and push them to your private ECR registry first. You can do this using the `fondant build` command.

But first you need to login into Docker with valid ECR credentials more info [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html):
```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
```

You will need to create a repository for you component first (one time operation):
```bash
 aws ecr create-repository --region <region> --repository-name <component_name>
```

Now you can use the `fondant build` [command](../components/publishing_components.md) (which uses Docker under the hood) to build and push your custom components to your private ECR registry:
```bash
fondant build <component dir> -t <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<component_name>:<tag>
```


#### Assigning custom resources to the pipeline

You can specify different kinds of `instance_type` while running your pipeline, you can see the available options [here](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html). You may need to request a [quota increase](https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html) for certain `instance_type`'s.