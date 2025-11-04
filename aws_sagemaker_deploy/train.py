import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_from_disk
from datasets import Dataset
from transformers import AutoConfig
from peft import PeftModel, PeftConfig
from pathlib import Path

from transformers.trainer_utils import is_main_process


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat", use_fast=True, trust_remote_code=True)
model = get_peft_model(model, lora_config)


tokenized_dataset = Dataset.load_from_disk("./qwen_chatml_dataset")


training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="/opt/ml/output/logs",
    save_steps=500,
    logging_steps=100,
)

print("Local rank:", training_args.local_rank)
print("Process rank:", training_args.process_rank)
print("World size:", training_args.world_size)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()


if is_main_process(training_args.local_rank):
    print("Merging LoRA into base model...")

    checkpoints = sorted(Path(training_args.output_dir).glob(
        "checkpoint-*"), key=os.path.getmtime)
    last_checkpoint = str(checkpoints[-1]) if checkpoints else None

    peft_config = PeftConfig.from_pretrained(last_checkpoint)

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, last_checkpoint)

    # Merge
    merged_model = model.merge_and_unload()

    # Save full merged model for SageMaker
    save_dir = "/opt/ml/model"  # SageMaker looks here
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("✅ Model saved for SageMaker deployment.")
