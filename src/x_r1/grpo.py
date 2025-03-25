# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


from configs import GRPOConfig
from rewards import (
    accuracy_reward,
    format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
)
from utils.callbacks import get_callbacks
from x_grpo_trainer import XGRPOTrainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from peft import LoraConfig, PeftModel, get_peft_model

#日志
logger = logging.getLogger(__name__)

import wandb


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.

    Args:
        training_args (object): 包含训练相关参数的对象，该对象应包含 `wandb_entity` 和 `wandb_project` 属性，
                                用于配置 Weights & Biases 服务。
    """
    # 检查 training_args 中是否指定了 wandb_entity
    # 如果指定了，则将其设置为环境变量 WANDB_ENTITY，供 Weights & Biases 服务使用
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    # 检查 training_args 中是否指定了 wandb_project
    # 如果指定了，则将其设置为环境变量 WANDB_PROJECT，供 Weights & Biases 服务使用
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project


#参数值
@dataclass
class GRPOScriptArguments(ScriptArguments):
    #奖励函数列表
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    #余弦相似度相关参数
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    #重复惩罚相关参数
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )


#定义推理模型交互格式
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    #为可重复性设置种子参数
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    #日志输出格式与方式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    #依据training_args的日志级别设置日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #设置transformers和datasets的日志级别
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    #记录每个进程的小摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #如果用户未指定从检查点恢复训练，则记录检查点路径
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    #如果用户选择使用 W&B 进行实验跟踪，则调用 init_wandb_training 函数设置相关环境变量
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # align the dataset
    #如果数据集是特定的医疗问题数据集，则重命名列以匹配模型输入格式
    if script_args.dataset_name == "FreedomIntelligence/medical-o1-verifiable-problem":
        dataset = dataset.rename_columns({
            "Open-ended Verifiable Question": "problem",
            "Ground-True Answer": "solution"
        })

    # Get reward functions
    #将奖励函数放在一个字典里
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
    }
    
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    #定义一个函数，将数据集中的每个样本格式化为对话形式。
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    #将格式化函数应用到数据集，将数据集中的每个样本转换为对话格式
    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 初始化模型参数
    logger.info("*** Initializing model kwargs ***")
    #设置参数数据类型
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    

    training_args.gradient_checkpointing = False
    model_kwargs = dict(
        revision=model_args.model_revision, # 模型版本
        trust_remote_code=model_args.trust_remote_code, # 是否信任远程代码如果设置为 True，则允许加载自定义的模型代码。
        attn_implementation=model_args.attn_implementation, # 注意力实现方式
        torch_dtype=torch_dtype, # torch数据类型
        use_cache=False if training_args.gradient_checkpointing else True, 
        #指定是否启用缓存机制。缓存可以加速推理，但在启用梯度检查点（gradient_checkpointing）时需要禁用缓存，以节省内存
    )

    # model = AutoModelForCausalLM.from_pretrained(**model_kwargs, pretrained_model_name_or_path = model_args.model_name_or_path)
    training_args.model_init_kwargs = model_kwargs
    # peft_config=get_peft_config(model_args)
    # print(peft_config)
    # if peft_config not None:
    #     model = get_peft_model(model, peft_config)
    # print(model)


    #############################
    # Initialize the XGRPO trainer
    #############################
    #初始化XGRPOTrainer
    trainer = XGRPOTrainer(
        model=model_args.model_name_or_path,#这里传入的是模型的路径或名称，通常是一个预训练模型的标识符（如 Hugging Face Hub 上的模型名称）。
                                            #如果是字符串，XGRPOTrainer 会根据路径加载模型；如果是已经实例化的模型对象，则直接使用。
        # model = model,
        reward_funcs=reward_funcs,#前面获取的奖励函数列表
        args=training_args,#是一个配置对象，包含训练过程中的各种参数，例如学习率、批量大小、梯度累积步数等
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        #如果 training_args.eval_strategy 设置为 "no"，则不使用验证数据集。
        peft_config=get_peft_config(model_args), # 一个 LoRA（Low-Rank Adaptation）配置对象，用于对模型进行参数高效微调。
        callbacks=get_callbacks(training_args, model_args),#回调函数列表，用于在训练过程中执行额外的逻辑（如日志记录、模型保存等）。
    )

    print(trainer)

    ###############
    # Training loop
    ###############
    #训练循环
    logger.info("*** Train ***")
    checkpoint = None
    #返回指定检查点
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    #返回最后一个检查点
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    #开始训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint) #传入检查点路径（如果有）
    #记录和保存训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    #保存模型
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["X-R1"],
    }
    if trainer.accelerator.is_main_process: #如果是主进程，保存模型卡
        #模型卡包含模型的关键信息（如数据集名称、标签等），便于分享和复现。
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True #恢复模型的缓存机制，以便进行快速推理。
        trainer.model.config.save_pretrained(training_args.output_dir) #保存模型配置

    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

#主函数
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    #解析命令行参数，生成 script_args、training_args 和 model_args。
    main(script_args, training_args, model_args )
