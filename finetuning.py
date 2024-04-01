import fire
import logging
import os
import torch
import torch.distributed as dist
import torch.optim as optim

from pkg_resources import packaging
from peft import get_peft_model, prepare_model_for_kbit_training

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    default_data_collator
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
)
from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    print_model_size,
    setup,
    setup_environ_flags,
    get_policies
)

from utils.data_processing import preprocess_dataset

logging.basicConfig(level=logging.INFO)


def update_kwargs(kwargs):
    kwargs['r'] = kwargs['lora_r']
    return kwargs


def main(**kwargs):
    if 'use_peft' in kwargs:
        kwargs = update_kwargs(kwargs)

    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

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

    # Load the tokenizer and add special tokens
    logging.info("Loading the tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    dataset_train, dataset_val = preprocess_dataset(kwargs, tokenizer)
    data_collator = default_data_collator

    if not train_config.enable_fsdp or rank == 0:
        logging.info(f"--> Training Set Length = {len(dataset_train)}")

    if not train_config.enable_fsdp or rank == 0:
        logging.info(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=data_collator,
        )

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
            # device_map="cuda",
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )

    logging.info("Printing Model Size")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        logging.info('Preparing the model for int8 training as quantization is enabled')
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        logging.info('Converting the model to bfloat16 as enable_fsdp and pure_bf16 = True')
        model.to(torch.bfloat16)

    if train_config.use_peft:
        logging.info('Using PEFT')
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        logging.info("HSDP device mesh is ready")

    # Setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        logging.info('Setting up FSDP if enable_fsdp is enabled')
        if not train_config.use_peft and train_config.freeze_layers:
            logging.info('Freezing transformer layers')
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if torch.cuda.is_available():
            model.to("cuda")

    # Initialize the optimizer and learning rate scheduler
    logging.info('Initializing the optimizer and learning rate scheduler')
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    logging.info('Starting the training process')
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )

    logging.info('Training process complete')

    if not train_config.enable_fsdp or rank == 0:
        for k, v in results.items():
            logging.info(f"Key: {k}, Value: {v}")


if __name__ == "__main__":
    fire.Fire(main)