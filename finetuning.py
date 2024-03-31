import fire
import logging
import os
import torch

from pkg_resources import packaging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
)
from llama_recipes.utils.train_utils import (
    print_model_size,
    setup,
    setup_environ_flags
)

logging.basicConfig(level=logging.INFO)


def update_kwargs(kwargs):
    kwargs['r'] = kwargs['lora_r']
    return kwargs


def main(**kwargs):
    kwargs = update_kwargs(kwargs)

    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # logging.info(f'TRAINING CONFIG: {train_config}')
    # logging.info(f'FSDP CONFIG: {fsdp_config}')

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        logging.info(f'Local rank is {local_rank}. Rank is {rank}. World Size is {world_size}')

    if torch.distributed.is_initialized():
        logging.info(f'Setting torch device = {rank}')
        torch.cuda.set_device(rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    logging.info('Loading the pre-trained model and setup its configuration')
    logging.info(f'Model Name: {train_config.model_name}')
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        logging.info('Both enable_fsdp and low_cpu_fsdp set to True')
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = AutoConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = AutoModelForCausalLM(llama_config)

    else:
        logging.info(f'enable_fsdp is set to {train_config.enable_fsdp} and low_cpu_fsdp is set to {train_config.low_cpu_fsdp}')
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )

    # Load the tokenizer and add special tokens
    logging.info("Loading the tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return


if __name__ == "__main__":
    fire.Fire(main)