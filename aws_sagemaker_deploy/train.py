from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from pathlib import Path
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

os.environ["HF_TOKEN"] = ""

output_dir = "/opt/ml/model"
save_dir = "gemma-lora-adapter"

base_model_id = "google/gemma-3-270m-it"

device = "cuda"


def get_dist_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


local_rank, rank, world_size = get_dist_info()

print(f"LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}")


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    # quantization_config=quantization_config if is_colab else None,
)

tokenizer.padding_side = "right"


ds = load_dataset("nbertagnolli/counsel-chat")

# drop null
ds = ds.filter(lambda x: x['questionText'] is not None)
ds = ds.filter(lambda x: x['topic'] is not None)


SYSTEM_PROMPT = "You are an assistant responsible for classifying mental health status."
MAX_LENGTH = 512
IGNORE_INDEX = -100


def build_chat(batch):
    questions = batch["questionText"]
    topics = batch["topic"]

    batch_messages = []
    batch_responses = []
    batch_conversations = []

    for q, t in zip(questions, topics):
        batch_messages.append([
            {"role": "user", "content": SYSTEM_PROMPT + "\n" + q},
        ])

        batch_responses.append(
            f"Based on what you've described, this sounds like '{t}'.")

        batch_conversations.append([
            {"role": "user", "content": SYSTEM_PROMPT + "\n" + q},
            {"role": "assistant",
                "content": f"Based on what you've described, this sounds like '{t}'."}
        ])

    return {
        "messages": batch_messages,
        "response": batch_responses,
        "conversation": batch_conversations
    }


chat_dataset = ds.map(build_chat, batched=True)


def format_conversations(batch):
    """
    Applies the model's chat template to a list of conversational turns.
    """
    texts = []
    for conversation in batch['conversation']:
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix('<bos>')
        texts.append(formatted_text)
    return {"text": texts}


formatted_dataset = chat_dataset.map(format_conversations, batched=True)

chat_dataset = ds.map(build_chat, batched=True)


def tokenize_and_mask_labels(batch):
    """
    Tokenizes the text and creates labels, masking the instruction/user tokens.
    """
    tokenized_results = []

    for conversation in batch['conversation']:
        full_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix('<bos>')

        prompt_conversation = conversation[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_conversation,
            tokenize=False,
            add_generation_prompt=True  # Ensures the model's response start token is included
        ).removeprefix('<bos>')

        full_tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        labels = full_tokenized["input_ids"].copy()

        prompt_length = len(prompt_tokenized["input_ids"])

        mask_end = min(prompt_length, len(labels))
        labels[:mask_end] = [IGNORE_INDEX] * mask_end

        input_ids = full_tokenized["input_ids"][:-1]
        labels = labels[1:]
        attention_mask = full_tokenized["attention_mask"][:-1]

        tokenized_results.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_tokenized["input_ids"],
            "prompt_attention_mask": prompt_tokenized["attention_mask"]
        })

    return {
        "input_ids": [r["input_ids"] for r in tokenized_results],
        "labels": [r["labels"] for r in tokenized_results],
        "attention_mask": [r["attention_mask"] for r in tokenized_results],
        "prompt_input_ids": [r["prompt_input_ids"] for r in tokenized_results],
        "prompt_attention_mask": [r["prompt_attention_mask"] for r in tokenized_results]
    }


tokenized_dataset = formatted_dataset.map(
    tokenize_and_mask_labels,
    batched=True,
    remove_columns=formatted_dataset["train"].column_names
)


def causal_lm_collator(batch):
    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]
    # prompt_input_ids = [x["prompt_input_ids"] for x in batch]
    # prompt_attention_mask = [x["prompt_attention_mask"] for x in batch]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    labels = pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        # "prompt_input_ids": prompt_input_ids,
        # "prompt_attention_mask": prompt_attention_mask,
    }


class LMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "prompt_input_ids": torch.tensor(item["prompt_input_ids"], dtype=torch.long),
            "prompt_attention_mask": torch.tensor(item["prompt_attention_mask"], dtype=torch.long),
        }


split_dataset = tokenized_dataset["train"].train_test_split(
    test_size=0.2, seed=42)

batch_size = 2

train_dataset = LMDataset(split_dataset["train"])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=causal_lm_collator,
)

val_dataset = LMDataset(split_dataset["test"])
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    # collate_fn=causal_lm_collator,
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

print("Preparing model for k-bit training...")
train_model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(train_model, lora_config)

peft_model.enable_input_require_grads()
peft_model.gradient_checkpointing_enable()
peft_model.config.use_cache = False

total_steps = len(train_dataloader) * 3


criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(
    peft_model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,  # ~5% of total steps
    num_training_steps=total_steps
)

scaler = GradScaler()


num_epochs = 3
accum_steps = 4
num_training_steps = num_epochs * len(train_dataloader) // accum_steps

print("Starting training...")

peft_model.train()
optimizer.zero_grad()

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        with autocast(device_type=device, dtype=torch.float16):
            outputs = peft_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = outputs.logits
            labels = batch["labels"].to(device)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            print(
                f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item() * accum_steps:.4f}")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    if (step + 1) % accum_steps != 0:
        print(
            f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item() * accum_steps:.4f}")

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

print("Training complete. Saving adapter model...")

peft_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Merging LoRA adapters with base model...")

tokenizer = AutoTokenizer.from_pretrained(save_dir)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, save_dir)

model.resize_token_embeddings(len(tokenizer))


merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")


# if is_main_process():
#     print("Starting training script...")
