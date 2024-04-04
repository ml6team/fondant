### SageMaker Runner

Leverage [AWS SageMaker](https://aws.amazon.com/sagemaker/) to run your Fondant workflow.

This makes it easy to scale up your workflows in a serverless manner without worrying about infrastructure
deployment.

The Fondant SageMaker runner will compile your Fondant workflow to a SageMaker pipeline spec and submit it to SageMaker.


!!! note "IMPORTANT"

    Using the SageMaker runner will create a [through cache rule](https://docs.aws.amazon.com/AmazonECR/latest/userguide/pull-through-cache.html) on the private ECR registry of your account. This is required to make sure that SageMaker can access the public [reusable images](../components/hub.md) used by Fondant components.

### Installing the SageMaker runner

Make sure to install Fondant with the SageMaker runner extra.

```bash
pip install fondant[sagemaker]
```

### Prerequisites
- You will need a sagemaker domain and user with the correct permissions. You can follow the instructions [here](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html) to set this up. Make sure to note down the role arn( `arn:aws:iam::<account_id>:role/service-role/AmazonSageMaker-ExecutionRole-<creation_timestamp>`) of the user you are using since you will need it.
- You will need to have an AWS account and have the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) installed and configured.
- Fondant on SageMaker uses an s3 bucket to store the pipeline artifacts. You will need to create an s3 bucket that SageMaker can use to store artifacts (manifests and data). You can create a bucket using the AWS CLI:

    ```bash
    aws s3 mb s3://<bucket-name>
    ```
!!! note "IMPORTANT"

    Regarding [the bucket and SageMaker permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-bucket.html):

    - If you use the the term 'sagemaker' in the name of the bucket, SageMaker will automatically have the correct permissions to the access bucket.
    - If you use any other name or existing bucket you will need to add a policy on the role that SageMaker uses to access the bucket. 


You can then set this bucket as the `base_path` of your pipeline with the syntax: `s3://<bucket_name>/<path>`.

### Running a pipeline with SageMaker


Since compiling a sagemaker spec requires access to the AWS SageMaker API, you will need to be logged in to 
AWS with a role that has all the required permissions to launch a SageMaker pipeline. 


=== "Console"
    
    ```bash 
    fondant run sagemaker <dataset_ref> \
     --working-dir $S3_BUCKET \
     --role-arn $SAGEMAKER_ROLE_ARN 
    ```
    

=== "Python"
    
    ```python
    from fondant.dataset.runner import SageMakerRunner
    
    runner = SageMakerRunner()
    runner.run(
        input=<path_to_dataset>,
        working_dir=<s3_bucket>,
        role_arn=<role_arn>,
        pipeline_name=<sagemaker_pipeline_name>
    )
    ```


Once your workflow is running you can monitor it using the SageMaker [Studio](https://aws.amazon.com/sagemaker/studio/).



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

The SageMaker runner supports assigning a specific `instance_type` to each component. This can be done by using the resources block when defining a component.

If not specified, the default `instance_type` is `ml.t3.medium`. The `instance_type` needs to be a valid SageMaker instance type you can find more info [here](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html).

```python
from fondant.dataset import Resources

images = raw_data.apply(
    "download_images",
    arguments={
        "input_partition_rows": 100,
        "resize_mode": "no",
    },
    resources=Resources(instance_type="ml.t3.xlarge"),
)
```