"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import os
import subprocess  # nosec
import tempfile
import argparse
import logging
import tarfile
from pathlib import Path
from typing import Union

from apscheduler.schedulers.background import BackgroundScheduler

# pylint: disable=import-error
import pyarrow.compute as pc
from google.cloud import storage
from sorcery import dict_of

from helpers.logger import get_logger
from helpers import storage_helpers, kfp_helpers, parquet_helpers
from helpers.manifest_helpers import DataManifest
from utils.dataset_utils import SDDatasetCreator
from utils import training_utils


def optional_int(passed_arg: any):
    """Custom function to handle optional integers in the argument parser"""
    return None if not passed_arg else int(passed_arg)


def parse_args():
    """Parse component arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", type=str, required=True, help="The run id of the pipeline"
    )
    parser.add_argument(
        "--artifact-bucket",
        type=str,
        required=True,
        help="The GCS bucket used to store the artifacts",
    )
    parser.add_argument(
        "--component-name", type=str, required=True, help="The name of the component"
    )
    parser.add_argument(
        "--project-id", type=str, required=True, help="The id of the gcp-project"
    )
    parser.add_argument(
        "--data-manifest-gcs-path",
        type=str,
        required=True,
        help="the path to the data manifest that contains information on the "
        "training set",
    )
    parser.add_argument(
        "--pretrained-model-gcs-path",
        type=str,
        required=True,
        help="Model identifier from hugginface.com/models",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        required=False,
        help="Whether training should be resumed from a previous checkpoint. Use a"
        "path saved in `pretrained_model_gcs_path` by `--checkpointing_steps`"
        "or `latest` to automatically select the last available checkpoint.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="The resolution for input images, all the images in the "
        "train/validation dataset will be resized to this resolution.",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        required=True,
        help="Batch size (per device) for the training dataloader",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        required=True,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--max-train-steps",
        type=optional_int,
        required=False,
        help="Total number of training steps to perform. If provided,"
        " overrides num_train_epochs",
    )
    parser.add_argument(
        "--checkpointing-steps",
        type=optional_int,
        required=False,
        help="Save a checkpoint of the training state every X updates."
        " These checkpoints are only suitable for resuming training"
        " using `--resume_from_checkpoint`.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        required=True,
        help="The number of updates steps to accumulate before performing a "
        "backward/update pass",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        type=str,
        required=False,
        help="Whether or not to use gradient checkpointing to save memory at"
        "the expense of slower backward pass",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale-lr",
        type=str,
        required=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation"
        " steps, and batch size",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        required=True,
        help='The scheduler type to use. Choose between ["linear", "cosine",'
        ' "cosine_with_restarts", "polynomial", "constant",'
        ' "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        required=True,
        help="Number of steps for the warmup in the lr scheduler",
    )
    parser.add_argument(
        "--use-ema", type=str, required=False, help="Whether to use EMA model"
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        required=True,
        help="Whether to use mixed precision. Choose between fp16 and bf16 "
        "(bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU."
        "Default to the value of accelerate config of the current system or"
        "the flag passed with the `accelerate.launch` command. Use this "
        "argument to override the accelerate config",
    )
    parser.add_argument(
        "--center-crop",
        type=str,
        required=False,
        help="whether to center crop images before resizing to resolution "
        "(if not set,random crop will be used)",
    )
    parser.add_argument(
        "--random-flip",
        type=str,
        required=False,
        help="whether to randomly flip the images horizontally",
    )
    parser.add_argument(
        "--model-gcs-path-finetuning-component",
        type=str,
        required=True,
        help="The path where the finetuned model and its checkpoints are saved",
    )

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
def sd_finetuning_component(
    run_id: str,
    artifact_bucket: str,
    component_name: str,
    project_id: str,
    data_manifest_gcs_path: str,
    pretrained_model_gcs_path: str,
    resume_from_checkpoint: Union[str, None],
    seed: int,
    resolution: int,
    train_batch_size: int,
    num_train_epochs: int,
    max_train_steps: int,
    checkpointing_steps: Union[int, None],
    gradient_accumulation_steps: int,
    gradient_checkpointing: Union[str, None],
    learning_rate: float,
    scale_lr: Union[str, None],
    lr_scheduler: str,
    lr_warmup_steps: int,
    use_ema: Union[str, None],
    mixed_precision: str,
    center_crop: Union[str, None],
    random_flip: Union[str, None],
    model_gcs_path_finetuning_component: str,
) -> None:
    """
    A component for finetuning a stable diffusion model on a custom dataset
    Args:
        run_id (str): the run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifacts
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        data_manifest_gcs_path (str): the path to the data manifest that contains information on the
        training set
        pretrained_model_gcs_path (str): Model identifier from huggingface.co/models or gcs
        path to a model
        resume_from_checkpoint (Union[str, None]): Whether training should be resumed from a
        previous checkpoint. Use a path saved in `pretrained_model_gcs_path` by
        `--checkpointing_steps`, or `"latest"` to automatically select the last available
         checkpoint.
        seed (int): A seed for reproducible training.
        resolution (int): The resolution for input images, all the images in the train/validation
        dataset will be resized to this resolution
        train_batch_size (int): Batch size (per device) for the training dataloader
        num_train_epochs (int): Total number of epochs to perform
        max_train_steps (int): Total number of training steps to perform. If provided overrides
         `num_train_epochs`
        checkpointing_steps (int): Save a checkpoint of the training state every X updates. These
        checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.
        gradient_accumulation_steps (int): The number of updates steps to accumulate before
         performing a backward/update pass
        gradient_checkpointing (str): Whether to use gradient checkpointing to save memory
        at the expense of slower backward pass
        learning_rate (float): Initial learning rate (after the potential warmup period) to use.
        scale_lr (str): Scale the learning rate by the number of GPUs, gradient accumulation steps,
         and batch size
        lr_warmup_steps (int): Scale the learning rate by the number of GPUs, gradient accumulation
        steps, and batch size
        lr_scheduler (str): The scheduler type to use. Choose between ["linear", "cosine",
         cosine_with_restarts", "polynomial", "constant","constant_with_warmup"]
        use_ema (str): Whether to use EMA model
        mixed_precision (str): Whether to use mixed precision. Choose between fp16 and bf16
         (bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU.
         Default to the value of accelerate config of the current system or the flag passed with the
        `accelerate.launch` command. Use this argument to override the `accelerate` config
        center_crop (str): whether to center crop images before resizing to resolution (if not set,
        random crop will be used)
        random_flip (str): whether to randomly flip images horizontally
        model_gcs_path_finetuning_component (str): The path where the finetuned model and its
         checkpoints are saved
    """

    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info("Started job...")

    # Show CUDA availability
    kfp_helpers.get_cuda_availability()

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download and load data manifest
        manifest_file_path = storage_helpers.download_file_from_bucket(
            storage_client, data_manifest_gcs_path, tmp_dir
        )
        file = tarfile.open(manifest_file_path, "r:gz")
        manifest_dict = file.extractfile(file.getmembers()[0]).read().decode("utf-8")
        data_manifest = DataManifest.from_json(manifest_dict)

        # Get relevant metadata information
        # Used to identify the dataset creation run associated with the training
        dataset_creation_run_id = data_manifest.metadata.run_id
        index_parquet_gcs_path = data_manifest.index
        datasets_parquet_dict = data_manifest.associated_data.dataset
        captions_parquet_dict = data_manifest.associated_data.caption

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition("-")[0]
        artifact_bucket_blob_path = (
            f"custom_artifact/{component_artifact_dir}/{component_name}"
        )
        logger.info(
            "custom artifact will be uploaded to %s",
            f"gs://{artifact_bucket}/{artifact_bucket_blob_path}",
        )

        # Initialize additional directories for temporary files
        imgs_tmp_path = os.path.join(tmp_dir, "img_dir")
        models_tmp_path = os.path.join(tmp_dir, "models")
        model_to_finetune_name = os.path.basename(pretrained_model_gcs_path)
        finetuned_model_name = (
            f"{model_to_finetune_name}-tmp_dir-{dataset_creation_run_id}"
        )
        finetuned_model_path = os.path.join(models_tmp_path, finetuned_model_name)
        model_to_finetune_path = os.path.join(models_tmp_path, model_to_finetune_name)
        finetuned_model_gcs_uri = (
            f"gs://{artifact_bucket}/{artifact_bucket_blob_path}/"
            f"{finetuned_model_name}"
        )
        os.makedirs(imgs_tmp_path, exist_ok=True)
        os.makedirs(finetuned_model_path, exist_ok=True)
        os.makedirs(model_to_finetune_path, exist_ok=True)

        # Download index files locally
        index_id_path = storage_helpers.download_file_from_bucket(
            storage_client, index_parquet_gcs_path, tmp_dir
        )

        # Get index list
        index_list = parquet_helpers.get_column_list_from_parquet(
            index_id_path, column_name="index"
        )
        filters = pc.field("file_id").isin(index_list)

        sd_dataset_creator = SDDatasetCreator(tmp_path=tmp_dir, img_path=imgs_tmp_path)

        for namespace in datasets_parquet_dict:
            # Download
            dataset_parquet_tmp_path = storage_helpers.download_file_from_bucket(
                storage_client, datasets_parquet_dict[namespace], tmp_dir
            )
            caption_parquet_tmp_path = storage_helpers.download_file_from_bucket(
                storage_client, captions_parquet_dict[namespace], tmp_dir
            )
            filtered_dataset = parquet_helpers.filter_parquet_file(
                file_path=dataset_parquet_tmp_path, filters=filters
            )
            filtered_captions = parquet_helpers.filter_parquet_file(
                file_path=caption_parquet_tmp_path, filters=filters
            )
            sd_dataset_creator.write_dataset(
                dataset_scanner=filtered_dataset,
                captions_scanner=filtered_captions,
                namespace=namespace,
            )

        # Download model to finetune (without checkpoints)
        training_utils.download_model_without_checkpoints(
            pretrained_model_gcs_path, finetuned_model_path
        )

        # Fetch specified checkpoint and download it to the local model path
        checkpoint_prefix = None
        if resume_from_checkpoint:
            # training script expects checkpoint to be located in the `output_dir` where the model
            # was previously trained -> download the checkpoints there
            checkpoint_prefix = training_utils.download_checkpoint_to_resume_from(
                resume_from_checkpoint=resume_from_checkpoint,
                pretrained_model_gcs_path=pretrained_model_gcs_path,
                finetuned_model_path=model_to_finetune_path,
            )

        # Initialize Daemon to upload model and checkpoints while training
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=training_utils.sync_training_checkpoints,
            trigger="interval",
            args=[model_to_finetune_path, finetuned_model_gcs_uri],
            minutes=10,
        )
        scheduler.start()

        # Specify training command (Subprocess requires integer values to be piped as string,
        # the argument parser of training script will convert them again to integers/floats)
        training_cmd = [
            "accelerate",
            "launch",
            "--config_file",
            "accelerate_config.yaml",
            "--mixed_precision",
            mixed_precision,
            "train_text_to_image.py",
            "--seed",
            str(seed),
            "--pretrained_model_name_or_path",
            finetuned_model_path,
            "--dataset_name",
            imgs_tmp_path,
            "--resolution",
            str(resolution),
            "--train_batch_size",
            str(train_batch_size),
            "--num_train_epochs",
            str(num_train_epochs),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--learning_rate",
            str(learning_rate),
            "--lr_scheduler",
            lr_scheduler,
            "--lr_warmup_steps",
            str(lr_warmup_steps),
            "--output_dir",
            model_to_finetune_path,
        ]

        # Add arguments defined with 'store_true' action if they are set to "True"
        optional_args = dict_of(
            random_flip, center_crop, use_ema, gradient_checkpointing, scale_lr
        )
        training_cmd.extend(
            [f"--{arg}" for arg, value in optional_args.items() if value]
        )

        # Add optional string/int/bool arguments
        if max_train_steps:
            training_cmd.extend(["--max_train_steps", str(max_train_steps)])
        if checkpointing_steps:
            training_cmd.extend(["--checkpointing_steps", str(checkpointing_steps)])
        if checkpoint_prefix:
            training_cmd.extend(["--resume_from_checkpoint", checkpoint_prefix])

        logger.info(f"Starting training: {training_cmd}")
        subprocess.run(training_cmd, check=True)  # nosec

        logger.info(
            "Training complete, waiting for model and checkpoint sync to upload the latest"
            " model state and shutting down gracefully ..."
        )
        scheduler.shutdown()

        # Run one last synchronization task to ensure that all files were uploaded partially (avoid
        # issues when scheduler is uploading partial files as they are being written)
        logger.info("Running final sync job")
        subprocess.run(
            [
                "gsutil",
                "-m",
                "rsync",
                "-r",
                finetuned_model_path,  # nosec
                finetuned_model_gcs_uri,
            ],
            check=True,
        )

        # Write manifest to outputPath
        Path(model_gcs_path_finetuning_component).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(model_gcs_path_finetuning_component).write_text(finetuned_model_gcs_uri)

        logger.info("Job completed.")


if __name__ == "__main__":
    args = parse_args()
    sd_finetuning_component(
        run_id=args.run_id,
        artifact_bucket=args.artifact_bucket,
        component_name=args.component_name,
        project_id=args.project_id,
        data_manifest_gcs_path=args.data_manifest_gcs_path,
        pretrained_model_gcs_path=args.pretrained_model_gcs_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        scale_lr=args.scale_lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        use_ema=args.use_ema,
        mixed_precision=args.mixed_precision,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        model_gcs_path_finetuning_component=args.model_gcs_path_finetuning_component,
    )
