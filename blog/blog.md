---
title: The Anatomy of LLM Fine-Tuning
date: 2025-12-10
slug: llm-fine-tuning-guide
updatedDate: 2025-12-27
description: A comprehensive deep dive into Large Language Model fine-tuning, covering advanced data strategies, parameter-efficient techniques (LoRA/QLoRA) and production deployment.
publish: true
featuredPost: true
tags:
  - python
  - deep-learning
  - llm
  - pytorch
categories:
  - Deep Learning
keywords:
    - llm
    - fine-tuning
    - deep learning
    - transfer learning
    - machine learning
    - llm fine-tuning
    - lora
    - qlora
    - llm evaluation
    - production ai
bannerImage: llm-fine-tuning.png
---

## Introduction

Imagine you’ve built an AI application powered by an LLM (Large Language Model) like **GPT-5** or **Gemini 3**. You demo it to your teammates, your manager, and your client; everyone loves it. You get the go-ahead to ship it to production.

A month later, the client returns with a new concern: the app is too expensive to operate. Inference costs are higher than expected, and they want you to reduce them. You think, *Maybe switching to a cheaper model will cut costs significantly.* You swap the model, test a set of real use cases, and the results look good enough. The client is happy again, and you deploy the update.

Three months later, you get another message. This time, the issue isn’t cost, it’s quality. The new model struggles with domain-specific queries, and the client wants better answers. You think, *Maybe I should fine-tune the model on data from this domain.* You gather examples, fine-tune, and validate the results, and the responses improve. Once again, the client approves the change, and you ship it.

That’s the core motivation behind fine-tuning: it lets you adapt a pre-trained model to your specific use case, often improving task performance and sometimes even reducing cost by enabling smaller models to perform well. In the following sections, we’ll walk through an end-to-end, practical notebook-style guide to fine-tuning LLMs.

<ExpandableSeeMore>

<Aside>

The diagrams in this post are built using [D2 Lang](https://d2lang.com/). It uses a simple declarative syntax to create diagrams. One can easily animate, change themes, and customize the diagrams inside the code editor itself. Here is the list of extensions for popular code editors: [Extensions](https://d2lang.com/tour/extensions/)  
Here is the code for the RAG diagram used in this post:

```d2
direction: down

**.style.border-radius: 8
**.style.font-color: "#333"

vector_store: "Vector Store" {
  shape: cylinder
}

documents: "Documents" {
  shape: document
  style: {
    multiple: true
  }
}

user_query: "User Query" {
  shape: person
}
llm: "Language Model" {
  shape: rectangle
}
top_k_chunks: "Top K Chunks" {
  style: {
    multiple: true
  }
}
response: "Response"

top_k_chunks -> llm: provide context
documents -> vector_store: create embeddings
vector_store -> top_k_chunks: retrieve top k
user_query -> vector_store: similarity search
user_query -> llm
llm -> response

```

</Aside>
</ExpandableSeeMore>

## What exactly is Fine-Tuning?

Fine-tuning is effectively a form of transfer learning. Instead of training a model from scratch (Pre-Training), which costs millions of dollars, we take a base model that already understands language structure and world knowledge, and we gently guide it towards a specific, specialized behavior.
Fine-tuning can be done using various techniques, such as full model fine-tuning, parameter-efficient fine-tuning (PEFT), which further includes methods like LoRA (Low-Rank Adaptation), and adapter-based fine-tuning.

Following is a simple diagram illustrating fine-tuning:

![Fine tuning](./diagrams/fine_tuning.svg)

## Comparison with RAG

You might be wondering: *Is fine-tuning the only way to adapt LLMs to specific tasks? Is there a simpler option?* Yes another popular approach is **Retrieval Augmented Generation (RAG)**.

RAG pairs a pre-trained language model with an external knowledge source (for example, a database or a document collection). At inference time, it retrieves relevant context and feeds it into the model so the output is grounded in that retrieved information without updating the model’s weights.

However, RAG isn’t always sufficient. If your task is highly specialized, or you need the model to follow a very specific style, structure, or decision pattern, retrieval alone may not get you there. That’s where fine-tuning helps: it can reliably change *how* the model responds.

In practice, teams often combine both: use RAG for up-to-date or proprietary knowledge, and fine-tune on a smaller domain dataset to shape behavior. The trade-off is extra system complexity (more components to build, operate, and maintain).

A common misconception is that fine-tuning adds *knowledge*. In practice:

* **RAG** adds **Knowledge**: "Here is the new policy document, answer based on it."
* **Fine-Tuning** adds **Skill/Behavior**: "Speak like a medical professional," "Output only valid JSON," or "Reason through this complex legal argument."

Following is a simple diagram illustrating RAG:

![RAG](./diagrams/rag.svg)

## Types of Fine-Tuning Techniques

There are several techniques for fine-tuning large language models, each with its own advantages and disadvantages. Some of the most common techniques include:

### Full Model Fine-Tuning

**Full model fine-tuning** involves updating all the parameters of the pre-trained model. This approach can lead to better performance but requires significant computational resources and large amounts of labelled data.In this case, all the weights of the model are updated during training.

The disadvantage of this approach is that it is computationally expensive and requires a lot of memory. It can also lead to overfitting if the dataset is small. It can also lead to catastrophic forgetting, where the model forgets the knowledge it learned during pre-training.

### PEFT

**PEFT (Parameter-Efficient Fine-Tuning)** involves updating only a small subset of the model's parameters, making it more efficient in terms of computation and memory usage. Techniques like LoRA (Low-Rank Adaptation) fall under this category. PEFT methods are particularly useful when dealing with large models and limited computational resources. They allow for faster training times and reduced memory consumption while still achieving good performance on specific tasks.

PEFT is further divided into multiple techniques like LoRA, QLoRA, Prefix Tuning, etc. Among these, LoRA has gained significant popularity due to its simplicity and effectiveness. LoRA works by introducing low-rank matrices into the model's architecture, allowing for efficient adaptation without modifying the original weights extensively.

## LoRA

**LoRA (Low Rank Adaptation)** is a parameter-efficient fine-tuning technique that introduces low-rank matrices into the model's architecture. Instead of updating all the weights of the model during fine-tuning, LoRA adds trainable low-rank matrices to certain layers of the model.

During training, only these low-rank matrices are updated, while the original weights remain frozen. This significantly reduces the number of parameters that need to be updated, making the fine-tuning process more efficient.

![LoRA](./diagrams/lora.svg)

&nbsp;

Here is the research paper on [LoRA: Low Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## QLoRA

**QLoRA (Quantized Low-Rank Adaptation)** is an extension of LoRA that combines low-rank adaptation with model quantization. In QLoRA, the pre-trained model is first quantized to reduce its memory footprint, and then LoRA is applied to fine-tune the quantized model. This approach allows for even more efficient fine-tuning, as the quantized model requires less memory and computational resources.

In other words, QLoRA enables fine-tuning of large language models on resource-constrained hardware, such as consumer-grade GPUs, by leveraging both quantization and low-rank adaptation techniques at a cost of minimal performance degradation.

>QLoRA takes more time as compared to LoRA, but consumes less memory during training as it uses quantization. QLoRA is suitable for scenarios where you have access to GPUs with limited memory. This is because during training, the model is dequantized to higher precision for computation, which can be time-consuming.

## Let's Fine-Tune a Model

In this blog, we will fine-tune a small instruction-tuned model using **LoRA** (a PEFT technique). We will be using pure PyTorch without high-level abstractions like Hugging Face [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) api to illustrate the core concepts and the internals of fine-tuning.

<Alert title='INFO' variant='info'>
For QLoRA, we have to load the model in 8-bit quantized format using `BitsAndBytesConfig` from `transformers` library. The rest of the code remains the same.

```python
from transformers.utils.quantization_config import BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
  model_id,
  quantization_config=BitsAndBytesConfig(load_in_8bit=True),
)
```

But make sure to load the model without quantization while merging LoRA adapters.
</Alert>

We will use **[google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-27b-it)** (Gemma 3, 270M parameters, instruction-tuned) model and **[nbertagnolli/counsel-chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat)** (Counsel Chat) dataset from Hugging Face.

The goal for this demo is intentionally simple: take a user message and have the model generate a short "classification-like" response (the dataset’s `topic`) in natural language.

<Alert title='WARNING' variant='warning'>
This dataset contains mental health-related text. Please do not use it for real medical advice or diagnosis, or in production systems.
</Alert>

## Companion Notebook

The code in this section mirrors the notebook [Gemma Fine-Tuning LoRA Google Colab](https://colab.research.google.com/github/gautamnaik1994/LLM_Fine_Tuning_Basic/blob/main/notebooks/Gemma_FineTuning.ipynb). Some parts are simplified for clarity.

The QLoRA version of this notebook is available [Gemma Fine-Tuning QLoRA Google Colab](https://colab.research.google.com/github/gautamnaik1994/LLM_Fine_Tuning_Basic/blob/main/notebooks/Gemma_FineTuning_QLoRA.ipynb).

## 1. Load Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
  attn_implementation="eager", # Gemma-specific optimization. Added after the model triggered a warning
)

tokenizer.padding_side = "right" # Found out that padding was on left by default, and fine-tuning required right padding.
```

The above code downloads the pre-trained Gemma model + tokenizer from Hugging Face.

### Sanity Check

Before you train anything, it’s worth seeing what the base model does on your task.

```python
import torch

messages = [
  {"role": "system", "content": "You are a assistant responsible for classifying mental health status."},
  {"role": "user", "content": "I am depressed and want to die"},
]

input_ids = tokenizer.apply_chat_template(
  messages,
  return_tensors="pt",
  add_generation_prompt=True,
).to(device)

attention_mask = torch.ones_like(input_ids).to(device)

outputs = model.generate(
  input_ids,
  attention_mask=attention_mask,
  max_new_tokens=100,
  do_sample=True,
  pad_token_id=tokenizer.eos_token_id,
  temperature=0.1,
)

input_len = input_ids.shape[1]
generated_tokens = outputs[0, input_len:]

print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
```

Following is a sample output from the above code:

```plaintext
I understand. It's very difficult to say what's happening, but I can offer some resources and support to help you cope. Please reach out to a mental health professional or a crisis hotline. You can also contact the National Suicide Prevention Services at 988 and the Crisis Text Line at 741740 in the US.
```

This baseline output gives you a reference point: after fine-tuning, you should see responses that look more like your training targets.

## 2. Load the Dataset

We will use the Counsel Chat dataset from Huggingface, which contains mental health-related questions and answers. You can load the dataset using the following code. The dataset is available [here](https://huggingface.co/datasets/counselchat/counselchat).

```python
from datasets import load_dataset

ds = load_dataset("nbertagnolli/counsel-chat")
```

**Quality vs. Quantity**

The above dataset contains around 2000 rows.

For instruction tuning, as few as **1,000 high-quality examples** can yield significant behavior changes. The [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) paper demonstrated that training on highly curated samples outperforms massive unfiltered scrapes.

## 3. Preprocess Dataset

We need to preprocess the dataset to convert it into a format suitable for training the Gemma model. The Gemma model expects the input in a specific format, so we will create a function to preprocess the data accordingly.

Following is a sample row from the dataset

```json
{
  "questionText": "I have so many issues to address. I have a history of sexual abuse, I’m a breast cancer survivor and I am a lifetime insomniac. I have a long history of depression and I’m beginning to have anxiety. I have low self esteem but I’ve been happily married for almost 35 years.\n  I’ve never had counseling about any of this. Do I have too many issues to address in counseling?",
  "topic": "depression",
}
```

Our initial goal is to format the data into a chat-like structure that Gemma understands.

### Why chat formatting matters

When fine-tuning instruction-tuned models like Gemma, the input format is crucial. These models are trained on specific conversation structures, and deviating from that can lead to suboptimal performance. That’s why
standardizing your data format is critical.

There are two common **Instruction Formats** used in the community:

* **Alpaca Format**: `{ "instruction": "...", "input": "...", "output": "..." }`
* **ShareGPT Format**: Multi-turn conversation history. This is generally superior for chat models as it teaches the model to handle context and follow-up questions.

Gemma is instruction-tuned and expects a particular chat syntax (special tokens, turn separators, etc.). The tokenizer’s `apply_chat_template(...)` is the reliable way to produce the *exact* text format the model was trained with.

Following is the chat template used in the notebook.
  
```jinja
<!-- print(tokenizer.get_chat_template()) -->

{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '
' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '
' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
```

For the sake of simplicity of explanation, we will only use the following data

```json
{
  "questionText": "I am depressed",
  "topic": "depression",
}
```

```python

SYSTEM_PROMPT = "You are an assistant."

messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n" + questionText},
            {"role": "assistant", "content": f"This sounds like '{topic}'."}
        ]
```

After formatting, the above example would look like this:

```xml
<bos><start_of_turn>user
You are a assistant.

I am depressed<end_of_turn>
<start_of_turn>model
```

But we cannot directly pass this text to the model. We need to convert this into tokens using the tokenizer. We will do that in the next step.

## 4. Supervised Fine-Tuning (SFT) Data Preparation & Loss Masking

Now that we have the conversations formatted, we need to tokenize them and create labels for training.

Fine-tuning here is **supervised fine-tuning (SFT)**: we show the model a full prompt + desired answer, then compute loss only on the answer tokens.

To do that, we build `labels` from `input_ids`, but we set prompt tokens to `-100` (the ignore index). PyTorch’s cross-entropy loss skips positions where the label is `-100`.

This is the most important step.

For the sake of better explanation, we have reformatted the data into the following structure:

```python
messages = {
    "messages": [
        [{"role": "user", "content": "You are an assistant.\nI am depressed"}]
    ],
    "response": ["This sounds like depression."],
    "conversation": [
        [
            {"role": "user", "content": "You are an assistant.\nI am depressed"},
            {"role": "assistant", "content": "This sounds like depression."},
        ]
    ],
}
```

After applying the chat template and tokenizing, we get the following token IDs:

```yaml
Full text:-
<start_of_turn>user
You are an assistant.
I am depressed<end_of_turn>
<start_of_turn>model
This sounds like depression.<end_of_turn>

Prompt text:-                
<start_of_turn>user
You are an assistant.
I am depressed<end_of_turn>
<start_of_turn>model

Full tokenized input_ids:     [2, 105, 2364, 107, 3048, 659, 614, 16326, 236761, 107, 236777, 1006, 41155, 106, 107, 105, 4368, 107, 2094, 12054, 1133, 17998, 236761, 106, 107]
Prompt tokenized input_ids:   [2, 105, 2364, 107, 3048, 659, 614, 16326, 236761, 107, 236777, 1006, 41155, 106, 107, 105, 4368, 107]
Labels:                       [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2094, 12054, 1133, 17998, 236761, 106, 107]
Input IDs shifted:            [2, 105, 2364, 107, 3048, 659, 614, 16326, 236761, 107, 236777, 1006, 41155, 106, 107, 105, 4368, 107, 2094, 12054, 1133, 17998, 236761, 106]
Labels shifted:               [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2094, 12054, 1133, 17998, 236761, 106, 107]
```

Look closely at the `labels`: the prompt tokens are `-100`, and only the response tokens have valid labels. This way, when we compute loss, only the response tokens contribute.
This means the model is only penalized for getting the response wrong, not the prompt.

Now look at the shifted `input_ids` and `labels`: they are shifted by one position. The last token in the `input_ids` has been removed. This is because, during training, the model predicts the next token given all previous tokens.

Now look closely at the shifted labels: The first `-100` has been removed. This is removed so that the labels align with the shifted `input_ids`. The model predicts the token at position `t` using the input tokens up to position `t-1`.

## 5. Efficient Batching with a Custom Collator

### Handling Variable Sequence Lengths  

While tokenizing, we ignored the length of sequences by setting `padding=False`. This means that each sequence can have a different length. However, for batching, we need to pad the sequences to the same length. The model requires inputs to be of the same length within a batch.
To handle this, we create a custom collator function that pads the `input_ids` and `labels` to the maximum length in the batch. We use `torch.nn.utils.rnn.pad_sequence` for padding.

### What is a collator?  

As a training dataset can be of huge size, it is not feasible to load the entire dataset into memory at once. Instead, we load the data in batches during training.  
Now, assume that we want to do some custom preprocessing on each batch just before feeding it to the model. This is where collators come into play. A collator is a function that takes a list of samples from the dataset and processes them into a batch. It can perform operations like padding, stacking, or any other custom preprocessing required for the model.

In our case, we avoided padding during tokenization to keep the dataset compact. Instead, we handle padding in the collator function.

This collator pads:

* `input_ids` with `tokenizer.pad_token_id`
* `labels` with `-100` (so padding tokens don’t contribute to loss)

```python
from torch.nn.utils.rnn import pad_sequence

def causal_lm_collator(batch):
  input_ids = [x["input_ids"] for x in batch]
  labels = [x["labels"] for x in batch]

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
  }
```

## 6. Set up LoRA

We will use the Hugging Face [PEFT](https://huggingface.co/docs/peft/en/index) library to set up LoRA for fine-tuning the Gemma model. First, we need to define the LoRA configuration using the `LoraConfig` class.
Then add target modules for LoRA adaptation. In Gemma, the attention projection layers are named `q_proj`, `k_proj`, `v_proj`, and `o_proj`.  
The target modules are the layers of the model that we want to fine-tune using LoRA.

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

> **Hyperparameter Intuition**:
>
> * **Rank (r)**: Determines the capacity of the adapter. Higher `r` means more parameters and potentially better performance, but higher memory usage. `r=8` is a standard starting point.
> * **Alpha**: Scales the learned weights. If you increase `r`, you usually increase `alpha` proportionally.

The above code sets up the LoRA configuration for fine-tuning the Gemma model. You can adjust the parameters based on your requirements. Then we wrap the pre-trained model with PEFT using the `get_peft_model` function.

```python
from peft import prepare_model_for_kbit_training

train_model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(train_model, lora_config)

peft_model.enable_input_require_grads()
peft_model.gradient_checkpointing_enable()
peft_model.config.use_cache = False # disable cache for gradient checkpointing

peft_model.print_trainable_parameters(); # trainable params: 737,280 || all params: 268,835,456 || trainable%: 0.2742
```

Note the `prepare_model_for_kbit_training` function. As per [Hugginface docs](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.prepare_model_for_kbit_training):

This method wraps the entire protocol for preparing a model before running a training. This includes:

* Cast the layernorm in fp32
* Making the output embedding layer require grads
* Add the upcasting of the LM head to fp32
* Freezing the base model layers to ensure they are not updated during training

In simpler terms, it prepares the model for efficient fine-tuning with LoRA by ensuring that only the necessary parts of the model are trainable and that numerical stability is maintained during training.

### Trainable Parameters

The output of `peft_model.print_trainable_parameters()` is as follows:

```plaintext
trainable params: 737,280 || all params: 268,835,456 || trainable%: 0.2742
```

This shows that only a small fraction of the model's parameters are trainable (**0.2742%** in this case). This is the key advantage of LoRA: it allows us to fine-tune large models with a small number of parameters, making the process more efficient.

### Model Structure with before LoRA

Before applying LoRA, the model looks like this:

```python
Gemma3ForCausalLM(
  (model): Gemma3TextModel(
    (embed_tokens): Gemma3TextScaledWordEmbedding(262144, 640, padding_idx=0)
    (layers): ModuleList(
      (0-17): 18 x Gemma3DecoderLayer(
        (self_attn): Gemma3Attention(
          (q_proj): Linear(in_features=640, out_features=1024, bias=False)
          (k_proj): Linear(in_features=640, out_features=256, bias=False)
          (v_proj): Linear(in_features=640, out_features=256, bias=False)
          (o_proj): Linear(in_features=1024, out_features=640, bias=False)
          (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
          (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
        )
        (mlp): Gemma3MLP(
          (gate_proj): Linear(in_features=640, out_features=2048, bias=False)
          (up_proj): Linear(in_features=640, out_features=2048, bias=False)
          (down_proj): Linear(in_features=2048, out_features=640, bias=False)
          (act_fn): GELUTanh()
        )
        (input_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (post_attention_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (pre_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (post_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
      )
    )
    (norm): Gemma3RMSNorm((640,), eps=1e-06)
    (rotary_emb): Gemma3RotaryEmbedding()
    (rotary_emb_local): Gemma3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=640, out_features=262144, bias=False)
)
```

### Model Structure after applying LoRA

What the model looks like after applying LoRA is shown below:

<ExpandableSeeMore>

```python
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): Gemma3ForCausalLM(
      (model): Gemma3TextModel(
        (embed_tokens): Gemma3TextScaledWordEmbedding(262144, 640, padding_idx=0)
        (layers): ModuleList(
          (0-17): 18 x Gemma3DecoderLayer(
            (self_attn): Gemma3Attention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=640, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=640, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=640, out_features=256, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=640, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=256, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=640, out_features=256, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=640, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=256, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=640, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=640, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
              (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
            )
            (mlp): Gemma3MLP(
              (gate_proj): Linear(in_features=640, out_features=2048, bias=False)
              (up_proj): Linear(in_features=640, out_features=2048, bias=False)
              (down_proj): Linear(in_features=2048, out_features=640, bias=False)
              (act_fn): GELUTanh()
            )
            (input_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
            (post_attention_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
            (pre_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
            (post_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
          )
        )
        (norm): Gemma3RMSNorm((640,), eps=1e-06)
        (rotary_emb): Gemma3RotaryEmbedding()
        (rotary_emb_local): Gemma3RotaryEmbedding()
      )
      (lm_head): Linear(in_features=640, out_features=262144, bias=False)
    )
  )
)
```

</ExpandableSeeMore>

As you can see, only the layers specified in `target_modules` are modified to include LoRA layers. The rest of the model remains unchanged. This is the key to parameter-efficient fine-tuning.

## 7. Training Loop

The notebook uses a manual training loop instead of Hugging Face `Trainer` api. This is great for learning because you can see exactly what is happening under the hood.

* forward pass -> logits
* compute loss with `ignore_index=-100`
* backprop with gradient accumulation
* optimizer + scheduler step

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)

total_steps = len(train_dataloader) * 3
scheduler = get_cosine_schedule_with_warmup(
  optimizer,
  num_warmup_steps=50,
  num_training_steps=total_steps,
)

scaler = GradScaler()
```

### Automatic Mixed Precision (AMP)

**What is `GradScaler` and `autocast`?**

`autocast` and `GradScaler` are the two main building blocks of PyTorch Automatic Mixed Precision (AMP). AMP improves training speed and reduces memory usage by running many operations in lower precision (typically `float16`/`bfloat16`) while keeping numerically sensitive parts in `float32`.

Mixed precision works best on hardware optimized for low-precision math (notably NVIDIA GPUs with Tensor Cores, and increasingly other backends such as AMD and Apple Silicon). The trade-off is that `float16` has a smaller dynamic range, so gradients can **underflow** (become 0) or **overflow** (become `inf`/`NaN`) more easily.

`GradScaler` solves this with **dynamic loss scaling**:

* It scales up the loss before backpropagation, which scales up gradients and helps prevent underflow.
* Before `optimizer.step()`, it unscales gradients back to their true magnitude so the update is correct.
* It automatically tunes the scale factor: if `inf`/`NaN` is detected, it lowers the scale for the next step; otherwise, it may gradually increase it for better utilization.

GradScaler doesn’t permanently "transform" the weights. It temporarily scales the loss (and therefore the gradients) to avoid float16 underflow, then unscales gradients back right before the optimizer updates the weights.

Numeric toy example (single weight)
Minimal PyTorch snippet demonstrating how `GradScaler` works:

```python
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast


w = torch.tensor([1.0], device="cuda", requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
scaler = GradScaler()

# Toy loss that produces a tiny gradient: loss = 1e-8 * w  ->  dloss/dw = 1e-8
with autocast(device_type="cuda", dtype=torch.float16):
    loss = (w * 1e-8).sum()

# backward on *scaled* loss (internally multiplies loss by scaler's scale)
scaler.scale(loss).backward()

# gradients are currently scaled (larger magnitude)
print("Scaled grad:", w.grad.item()) # 0.0006553599960170686

# step() will unscale grads, check inf/nan, then call opt.step() safely
scaler.step(opt)
scaler.update()
opt.zero_grad()

print("Updated w:", w.item())  # Should be 0.999999999 but will get 1.0 as the change is below float32/float16 display precision
```

Following is the full training loop with gradient accumulation:

```python
num_epochs = 3
accum_steps = 4
num_training_steps = num_epochs * len(train_dataloader) // accum_steps

progress_bar = tqdm(range(num_training_steps))

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
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item() * accum_steps:.4f}")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            progress_bar.update(1)

    if (step + 1) % accum_steps != 0:
        print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item() * accum_steps:.4f}")

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        progress_bar.update(1)
```

### Gradient Accumulation

When I prototyped this on my MacBook, a batch size of **16** with a sequence length of **1024** worked fine. Running the exact same setup on Google Colab immediately triggered an out-of-memory (OOM) error simply because the available GPU memory was lower.

#### Who consumes GPU memory?

**Model parameters**

* Stored once
* Fixed cost

**Optimizer states**

* Example: Adam uses ~2x parameter size
* Also fixed per model

**Gradients**

* Same size as model parameters
* Stored after backward pass

**Activations (dominant factor)**

* Intermediate tensors from the forward pass
* Required for backpropagation
* Scales linearly with batch size and sequence length

#### Memory usage formula (approximate)

```plaintext
Total Memory ~= Model Params + Optimizer States + Gradients
            + (Batch Size x Sequence Length x Activation Size)
```

In practice, the activation term is often the dominant one during training, so increasing batch size or sequence length can push you over the VRAM limit quickly.

#### What can we do about it?

The obvious fix is to reduce the batch size (because activations scale with it). But training with very small batches has downsides:

* Highly noisy gradients
* Frequent direction changes of gradient descent vector
* Requires a smaller learning rate
* More training steps to converge

This is where **gradient accumulation** comes in (see `accum_steps` in the training loop).

In a standard training loop, every batch does:
`loss.backward()` -> `optimizer.step()` -> `optimizer.zero_grad()`

With gradient accumulation, you split a "large" batch into multiple **micro-batches** that fit in memory. You run `loss.backward()` on each micro-batch (accumulating gradients), and only after `accum_steps` micro-batches do you update weights with `optimizer.step()` and then clear gradients.

In PyTorch, gradients accumulate by default: parameters with `requires_grad=True` store gradients in their `.grad` buffer until you explicitly clear them with `optimizer.zero_grad()`.

```plaintext
Without gradient accumulation:
  Batch = 2 -> update
  Batch = 2 -> update
  Batch = 2 -> update
  Batch = 2 -> update

With gradient accumulation (accum_steps = 4):
  Batch = 2 -> accumulate
  Batch = 2 -> accumulate
  Batch = 2 -> accumulate
  Batch = 2 -> update
```

The net effect is a larger **effective batch size** without needing to fit it all in memory at once:

```plaintext
effective_batch_size = micro_batch_size × accum_steps
```

So if your GPU can only fit a micro-batch of 2, using `accum_steps=4` behaves similarly (in terms of gradient signal) to training with batch size 8, while keeping memory usage closer to batch size 2.

## 8. Save the LoRA Adapter

With LoRA, you typically save just the adapter weights (small), not the entire base model (large).

```python
save_dir = "gemma-lora-adapter"
peft_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

## 9. Run inference

To run inference, you load the original base model and then attach the adapter.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_id = "google/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    attn_implementation="eager",
)

model = PeftModel.from_pretrained(base_model, save_dir)

print("Tokenizer vocab size:", len(tokenizer)) # Tokenizer vocab size: 262145
print("Model embedding size:", model.get_input_embeddings().weight.shape[0]) # Model embedding size: 262144
```

### Debugging Model/Tokenizer Vocab Mismatch

Notice that the tokenizer vocab size (262145) is one more than the model embedding size (262144).
This led to a CUDA error, To fix this, we can resize the model embeddings to match the tokenizer vocab size by calling `model.resize_token_embeddings(len(tokenizer))`.
This happens when model may have added any special token to the embedding layer which might not be part of orginal model's vocab.

```python
if device == "mps":
    model.to("cpu")
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval();

print("Tokenizer vocab size:", len(tokenizer)) # Tokenizer vocab size: 262145
print("Model embedding size:", model.get_input_embeddings().weight.shape[0]) # Model embedding size: 262145
```

```python
messages = [
    {"role": "user", "content": "You are a assistant responsible for classifying mental health status. I am bored and sad"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,).to(device)
attention_mask = torch.ones_like(input_ids).to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
    )

input_len = input_ids.shape[1]
generated_tokens_tensor = outputs[0, input_len:]
decoded_response = tokenizer.decode(generated_tokens_tensor, skip_special_tokens=True)

print(decoded_response) # Based on what you've described, this sounds like 'depression'.
```

## 10. Merge the Adapter

Adapters are great for iteration (small artifacts). For deployment, you sometimes want a single merged model directory.

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./gemma-merged")
tokenizer.save_pretrained("./gemma-merged")
```

Then load the merged model for inference:

```python
tuned_model = AutoModelForCausalLM.from_pretrained("./gemma-merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./gemma-merged")

messages = [
    {"role": "user", "content": "You are a assistant responsible for classifying mental health status. I feel anxious and stressed all the time."}
]


input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
attention_mask = torch.ones_like(input_ids).to(device)

outputs = tuned_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.1,
)

input_len = input_ids.shape[1]
generated_tokens_tensor = outputs[0, input_len:]
decoded_response = tokenizer.decode(generated_tokens_tensor, skip_special_tokens=True)

print(decoded_response) # Based on what you've described, this sounds like 'stress'.
```

## Evaluation & Metrics

How do we know the model is actually "better"? Loss going down is necessary but not sufficient.

### 1. Perplexity

Perplexity measures how "surprised" the model is by the next token. Lower is **normally** better, but a model can have low perplexity and still hallucinate or be unsafe.

### 2. LLM-as-a-Judge

The industry standard for evaluation is now using a stronger model (e.g., GPT-4o) to grade the responses of your fine-tuned model.
We feed the prompt, the gold reference answer, and our model's answer to the "Judge LLM" and ask it to score alignment on a 1-5 scale.

```python
judge_prompt = """
Review the Assistant's answer compared to the Reference.
Rank the quality on a scale of 1-5 based on correctness and tone.
User Query: {query}
Reference: {reference}
Assistant: {prediction}
"""
```

### 3. Custom Evaluation Logic

In my fine-tuning example, I used a straightforward approach: I compared the mental health topic predicted by the model to the reference topic in the dataset. If they matched, the prediction was marked as correct.

## Model Performance Improvement

Before fine tuning, following were the results:

```plaintext
I understand. It's very difficult to say what's happening, but I can offer some resources and support to help you cope. Please reach out to a mental health professional or a crisis hotline. You can also contact the National Suicide Prevention Services at 988 and the Crisis Text Line at 741740 in the US.
```

After fine tuning, we can see that the output is streamlined and as per our expectations

```plaintext
Based on what you've described, this sounds like 'stress'.
```

## Deployment on AWS SageMaker

SageMaker is a good option when you want to run training as a managed job (on a GPU instance) and then host the fine-tuned model behind a managed HTTPS endpoint.

### 1. Set up the SageMaker environment

* Create a SageMaker Notebook (or Studio) environment to *orchestrate* training and deployment. A cheap CPU instance is enough here because it mainly submits jobs.
* Install a compatible SDK version. As of today, pin:

  `pip install sagemaker==2.255.0`

  (SageMaker SDK v3 does not support the Hugging Face integration used below.)

### 2. Create a training job

We create the training job using the SageMaker Hugging Face estimator. The training script lives in [train.py](https://github.com/gautamnaik1994/LLM_Fine_Tuning_Basic/blob/main/aws_sagemaker_deploy/train.py). Put this file in a folder named `src` alongside your notebook.

The complete working notebook is here: [Gemma Fine-Tuning LoRA on SageMaker](https://colab.research.google.com/github/gautamnaik1994/LLM_Fine_Tuning_Basic/blob/main/aws_sagemaker_deploy/main.ipynb)

```python
import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

role = sagemaker.get_execution_role()

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='src',
    instance_type='ml.g5.4xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.56.2',
    pytorch_version='2.8.0',
    py_version='py312',
    base_job_name='finetune-llm',
    hyperparameters={
    }
)

huggingface_estimator.fit()
```

### 3. Deploy the fine-tuned model

After training finishes, SageMaker writes model artifacts to S3. You can then create a Hugging Face model and deploy it as a real-time endpoint:

```python
huggingface_model = HuggingFaceModel(
    model_data=huggingface_estimator.model_data,  # output of huggingface_estimator.model_data
    role=role,
     transformers_version='4.51.3',
    pytorch_version='2.6.0',
    py_version='py312',
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.2xlarge'
)
```

### 4. Run inference

Once the endpoint is up, you can call it with the same chat-formatted prompt you used during fine-tuning:

```python
prompt = """
<bos><start_of_turn>user
You are an assistant responsible for classifying mental health status.
I feel anxious and stressed all the time.<end_of_turn>
<start_of_turn>model
"""

response = predictor.predict({
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 100,
        "do_sample": False,
    }
})

generated = response[0]['generated_text']
print("Assistant:", generated)
```

## Github Repository

The complete code for this blog is available in my GitHub repository: [LLM_Fine_Tuning](https://github.com/gautamnaik1994/LLM_Fine_Tuning_Basic)

## Conclusion

Fine-tuning is a key tool in an AI engineer’s toolkit, enabling the transition from general-purpose language understanding to strong, domain-specific performance. By combining data-centric engineering, efficient training methods like LoRA, and robust evaluation workflows, we can build AI systems that are not only capable but also aligned with real business requirements.

The future of applied AI will be driven less by ever-larger models and more by specialized, efficient, and well-engineered systems that tightly integrate models with high-quality data.

As future work, I plan to implement distributed training to scale fine-tuning workflows more effectively and also explore reinforcement learning from human feedback (RLHF) to further align model outputs with user expectations.
