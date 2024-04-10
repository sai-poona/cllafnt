import logging
import subprocess
from typing import List

import torch

import boto3
import tarfile
import os
import time
from urllib.parse import urlparse


def get_bucket_and_key(s3_uri):
    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != 's3':
        raise ValueError("Not an S3 URI")
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    return bucket, key


def tar_and_upload_to_s3(
    folder_path,
    s3_uri="s3://sandbox-dump/codellama-test/finetuned_models",
    sns_topic_arn="arn:aws:sns:us-east-1:588907522565:training-codellama"
):
    """
    Tars a folder, uploads the tar file to S3, and sends a notification via email based on the success or failure of the upload.

    :param folder_path: Path to the folder to tar and upload.
    :param bucket_name: Name of the S3 bucket.
    :param sns_topic_arn: ARN of the SNS topic to send notifications.
    :return: True if tar and upload were successful and notification sent, False otherwise.
    """
    # Generate timestamp for the tar file
    logging.info("Model Tarring process has begun")
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    s3_folder_prefix = '_'.join(folder_path.split("/")[-2:])
    # tar_file_name = f"folder_{timestamp}.tar.gz"  
    tar_file_name = f"{s3_folder_prefix}_{timestamp}.tar.gz"  
    # Create a tar file from the folder
    with tarfile.open(tar_file_name, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    logging.info("Uploading to S3")
    # Upload the tar file to S3
    s3_client = boto3.client('s3')
    sns_client = boto3.client('sns')
    try:
        bucket, key = get_bucket_and_key(s3_uri)
        s3_client.upload_file(tar_file_name, bucket, f"{key}/{tar_file_name}")
        logging.info(f"Uploaded {tar_file_name} to S3 bucket {s3_uri}")
        # Send success notification via email
        message = f"Tar file {tar_file_name} uploaded successfully to S3"
        subject = "S3 File Upload Notification - Success"
        sns_client.publish(TopicArn=sns_topic_arn, Message=message, Subject=subject)
        logging.info("Success notification sent via email!")
        return True
    except Exception as e:
        logging.error(f"Error uploading {tar_file_name} to S3: {e}")
        # Send failure notification via email
        message = f"Error uploading {tar_file_name} to S3: {e}"
        subject = "S3 File Upload Notification - Failure"
        sns_client.publish(TopicArn=sns_topic_arn, Message=message, Subject=subject)
        logging.error("Failure notification sent via email!")
        return False
    finally:
        # Clean up: Remove the tar file
        os.remove(tar_file_name)
        # print(0)
                        
def get_num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0


def run_with_error_handling(command: List[str], shell=False):
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess script failed with return code: {e.returncode}")
        raise RuntimeError(e)