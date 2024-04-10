import argparse


def str2bool(v: str) -> bool:
    """Convert string argument to a boolean value."""
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--batching_strategy', type=str, default='packing')
    parser.add_argument('--context_length', type=int, default=4096)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=str2bool, default=False)
    parser.add_argument('--gradient_clipping_threshold', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_workers_dataloader', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--int8_quantization', type=str2bool, default=False)
    parser.add_argument('--freeze_layers', type=str2bool, default=False)
    parser.add_argument('--num_freeze_layers', type=int, default=1)
    parser.add_argument('--use_fast_kernels', type=str2bool, default=False)
    parser.add_argument('--save_metrics', type=str2bool, default=False)
    parser.add_argument('--run_validation', type=str2bool, default=True)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--enable_fsdp', type=str2bool, default=False)
    parser.add_argument('--fsdp_checkpoint_root_dir', type=str, default='checkpoints')
    parser.add_argument('--low_cpu_fsdp', type=str2bool, default=False)
    parser.add_argument('--mixed_precision', type=str2bool, default=True)
    parser.add_argument('--use_fp16', type=str2bool, default=False)
    parser.add_argument('--pure_bf16', type=str2bool, default=False)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--save_optimizer', type=str2bool, default=False)
    parser.add_argument('--use_peft', type=str2bool, default=True)
    parser.add_argument('--peft_method', type=str, default='lora')
    parser.add_argument('--peft_output_dir', type=str, default='peft_model')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--train_dir", type=str, default="./")
    parser.add_argument("--validation_dir", type=str, default="./")
    parser.add_argument("--file_extension", type=str, default="jsonl")
    parser.add_argument("--prompt_template", type=str, default="template.json")
    parser.add_argument("--validation_split_ratio", type=float, default=0.2)
    parser.add_argument("--max_input_length", type=int, default=-1)
    parser.add_argument('--model_output_dir', type=str, default='finetuned_model')

    return parser.parse_known_args()