import fire
import glob
import logging
import os
import shutil
import tarfile
import torch


from config.args import _parse_args
from utils.base import get_num_gpus, run_with_error_handling

from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

logging.basicConfig(level=logging.INFO)

LLAMA_RECIPES_FOLDER = "llama_recipes"
LLAMA_RECIPES_TARBALL = "llama_recipes.tar.gz"

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            logging.info(f"Folder '{folder_path}' and all its subfolders have been deleted.")
        except OSError as e:
            logging.error(f"Error: {folder_path} : {e.strerror}")
    else:
        logging.info(f"Folder '{folder_path}' does not exist.")


def untar_llama_finetuning_recipe_tarball(tarball_path: str, target: str) -> None:
    """Untar the LLama Finetuning receipe repo."""

    delete_folder(LLAMA_RECIPES_FOLDER)

    logging.info("Untarring Llama Recipes Tarball")
    with tarfile.open(tarball_path, "r") as llama_recipe_tar:
        llama_recipe_tar.extractall(target)

    logging.info("Untar Complete")


def merge_base_and_peft_model(base_model_dir: str, peft_model_dir: str, save_model_dir: str) -> None:
    """Load the pre-trained base model and the adapter peft module, merge them and save it to the disk."""
    logging.info("Combining pre-trained base model with the PEFT adapter module.")
    model_to_merge = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16), peft_model_dir
    )

    merged_model = model_to_merge.merge_and_unload()
    logging.info("Saving the combined model in safetensors format.")
    merged_model.save_pretrained(save_model_dir, safe_serialization=True)
    logging.info("Saving complete.")

    logging.info("Saving the tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.save_pretrained(save_model_dir)
    logging.info("Saving complete.")



def create_invoking_command(args):
    try:
        num_gpus = get_num_gpus()
        if args.enable_fsdp:
            command = [
                'torchrun',
                '--nnodes',
                '1',
                '--nproc_per_node',
                str(num_gpus),
            ]
        else:
            command = ['python']

        command += [
            'finetuning.py',
            '--num_gpus',
            str(num_gpus),
            '--model_name',
            f'{args.model_dir}',
            '--batch_size_training',
            f'{args.per_device_train_batch_size}',
            '--batching_strategy', #
            f'{args.batching_strategy}',
            '--context_length', #
            f'{args.context_length}',
            '--gradient_accumulation_steps', #
            f'{args.gradient_accumulation_steps}',
            '--gradient_clipping', #
            f'{args.gradient_clipping}',
            '--gradient_clipping_threshold', #
            f'{args.gradient_clipping_threshold}',
            '--num_epochs',
            f'{args.num_epochs}',
            '--num_workers_dataloader',
            f'{args.num_workers_dataloader}',
            '--lr',
            f'{args.learning_rate}',
            '--weight_decay',
            f'{args.weight_decay}',
            '--gamma', #
            f'{args.gamma}',
            '--seed',
            f'{args.seed}',
            '--freeze_layers', #
            f'{args.freeze_layers}',
            '--num_freeze_layers', #
            f'{args.num_freeze_layers}',
            '--use_fast_kernels', #
            f'{args.use_fast_kernels}',
            '--save_metrics', #
            f'{args.save_metrics}',
            '--run_validation', #
            f'{args.run_validation}',
            '--val_batch_size', #
            f'{args.val_batch_size}',
            '--quantization',
            f'{args.int8_quantization}',
            '--train_dir',
            f'{args.train_dir}',
            '--validation_dir',
            f'{args.validation_dir}',
            '--file_extension',
            f'{args.file_extension}',
            '--prompt_template',
            f'{args.prompt_template}',
            '--validation_split_ratio',
            f'{args.validation_split_ratio}',
            '--max_input_length',
            f'{args.max_input_length}',
        ]

        if args.enable_fsdp:
            command += [
                '--enable_fsdp',
                '--dist_checkpoint_root_folder',
                f'{args.fsdp_checkpoint_root_dir}',
                '--low_cpu_fsdp', #
                f'{args.low_cpu_fsdp}',
                '--mixed_precision', #
                f'{args.mixed_precision}',
                '--use_fp16', #
                f'{args.use_fp16}',
                '--pure_bf16',
                f'{args.pure_bf16}',
                '--optimizer', #
                f'{args.optimizer}',
                '--save_optimizer', #
                f'{args.save_optimizer}',
            ]

        if args.use_peft:
            command += [
                '--use_peft',
                '--peft_method',
                f'{args.peft_method}',
                '--output_dir',
                f'{args.peft_output_dir}',
                '--lora_r',
                f'{args.lora_r}',
                '--lora_alpha',
                f'{args.lora_alpha}',
                '--lora_dropout',
                f'{args.lora_dropout}',
                '--target_modules',
                f'{args.target_modules}',
            ]
        return command
    except Exception as e:
        logging.error(f'Error creating invoking command: {e}')
        return


def main(**kwargs):
    # untar_llama_finetuning_recipe_tarball(tarball_path=LLAMA_RECIPES_TARBALL, target=".")
    finetuning_args, _ = _parse_args()
    logging.info(f'Finetuning Args: {finetuning_args}')
    command = create_invoking_command(finetuning_args)
    if command is not None:
        logging.info("Executing command:")
        logging.info(command)
        # run_with_error_handling(command)

        if finetuning_args.use_peft:
            merge_base_and_peft_model(
                base_model_dir=finetuning_args.model_dir,
                peft_model_dir=finetuning_args.peft_output_dir,
                save_model_dir=finetuning_args.model_output_dir,
            )

            for filename in glob.glob(os.path.join(os.path.join(finetuning_args.peft_output_dir, "tokenizer"), "*.*")):
                print(filename)
                shutil.copy(filename, finetuning_args.model_output_dir)


if __name__ == "__main__":
    fire.Fire(main)