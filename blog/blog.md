---
title: LLM Fine Tuning
date: 2025-12-10
slug: llm-fine-tuning
updatedDate: 2025-12-10
description: This case study delves deeply into LLM fine-tuning to uncover insights
  about the process and assist in recommending best practices for effective
  fine-tuning.
publish: true
featuredPost: false
tags:
  - python
categories:
  - Deep Learning
keywords:
    - llm
    - fine-tuning
    - deep learning
    - transfer learning
    - machine learning
bannerImage: llm-fine-tuning.png
---

## Introduction

Assume that you built a kickass AI application using a large language model (LLM) like GPT-4 or LLaMA. You showed it to your teammates, your manager, your client. Everyone is happy. You get the go ahead to deploy it to production.

After 1 month, your client comes back to you. He is not happy with the cost of running the application. The inference cost is too high. Instead of saving cost, the application is costing more money than before. He wants you to reduce the cost of running the application.
You think to yourself, "Hmm, maybe I can fine-tune a smaller model on my specific use case. That should reduce the inference cost significantly". You replace the model, test it with some use cases, and the results were satisfactory. Your client is happy again. You get the go ahead to deploy the model to production.

After 3 months, your client comes back to you again. This time, he is not happy with the quality of the responses from the model. The model is not able to handle some specific queries related to his domain. He wants you to improve the quality of the responses. You think to yourself, "Hmm, maybe I can fine-tune the model further on more specific data related to his domain". You gather more data, fine-tune the model again, test it with some use cases, and the results were satisfactory. Your client is happy again. You get the go ahead to deploy the model to production.

Above is an example of how fine-tuning can help improve the performance and reduce the cost of running an AI application. Fine-tuning allows you to adapt a pre-trained model to your specific use case, making it more efficient and effective. In the following sections, we will explore how to fine-tune large language models (LLMs). We will also discuss how to deploy the fine-tuned models using AWS SageMaker.

## What exactly is Fine-Tuning?

![alt text](./diagrams/fine_tuning.svg)

Fine-tuning is the process of taking a pre-trained model and training it further on a specific dataset to adapt it to a particular task or domain. This involves updating the model's weights based on the new data while retaining the knowledge learned during the initial pre-training phase. Fine-tuning can be done using various techniques, such as full model fine-tuning, parameter-efficient fine-tuning (PEFT) which further includes methods like LoRA (Low-Rank Adaptation), and adapter-based fine-tuning.

## Comparison with RAG

![alt text](./diagrams/rag.svg)

You might be wondering, "Is fine tuning the only way to adapt LLMs to specific tasks? Is there any other simpler way?" The answer is yes. Another popular approach is Retrieval-Augmented Generation (RAG). RAG combines pre-trained language models with external knowledge sources, such as databases or document collections, to enhance the model's ability to generate relevant and accurate responses. Instead of fine-tuning the entire model, RAG retrieves relevant information from the external source and incorporates it into the generation process.

But here is the minor observation, Although RAG is simpler and preferable in many scenarios, it may not always be sufficient for highly specialized tasks that require deep domain knowledge or specific language patterns. Sometime you might want the model to reply in a specific way that is only possible through fine-tuning. In such cases, fine-tuning becomes necessary to achieve the desired performance.

The ideal approach often involves a combination of both fine-tuning and RAG, where the model is fine-tuned on a smaller, domain-specific dataset while also leveraging external knowledge sources for enhanced performance. This again leads to multiple issues, like need for using and maintaining multiple systems, increased complexity, etc. So, it is always a trade-off between simplicity and performance.

## Types of Fine-Tuning Techniques

There are several techniques for fine-tuning large language models, each with its own advantages and disadvantages. Some of the most common techniques include:

### Full Model Fine-Tuning

Involves updating all the parameters of the pre-trained model. This approach can lead to better performance but requires significant computational resources and large amounts of labeled data.
In this case all the weights of the model are updated during training. The disadvantage of this approach is that it is computationally expensive and requires a lot of memory. It can also lead to overfitting if the dataset is small. It can also lead to catastrophic forgetting, where the model forgets the knowledge it learned during pre-training.

### Parameter-Efficient Fine-Tuning (PEFT)

Involves updating only a small subset of the model's parameters, making it more efficient in terms of computation and memory usage. Techniques like LoRA (Low-Rank Adaptation) fall under this category. PEFT methods are particularly useful when dealing with large models and limited computational resources. They allow for faster training times and reduced memory consumption while still achieving good performance on specific tasks.

PEFT is futher divided into multiple techniques like LoRA, Adapters, Prefix Tuning, etc. Among these, LoRA has gained significant popularity due to its simplicity and effectiveness. LoRA works by introducing low-rank matrices into the model's architecture, allowing for efficient adaptation without modifying the original weights extensively.

## LoRA (Low-Rank Adaptation)

![alt text](./diagrams/lora.svg)

LoRA is a parameter-efficient fine-tuning technique that introduces low-rank matrices into the model's architecture. Instead of updating all the weights of the model during fine-tuning, LoRA adds trainable low-rank matrices to certain layers of the model. During training, only these low-rank matrices are updated, while the original weights remain frozen. This significantly reduces the number of parameters that need to be updated, making the fine-tuning process more efficient.

LoRA takes less time as compared to QLoRA, but consumes more memory during training as it does not use quantization. LoRA is suitable for scenarios where you have access to GPUs with sufficient memory to handle the model size.

The research paper introducing LoRA can be found [here](https://arxiv.org/abs/2106.09685).

## QLoRA (Quantized Low-Rank Adaptation)

QLoRA is an extension of LoRA that combines low-rank adaptation with model quantization. In QLoRA, the pre-trained model is first quantized to reduce its memory footprint, and then LoRA is applied to fine-tune the quantized model. This approach allows for even more efficient fine-tuning, as the quantized model requires less memory and computational resources.
In other words, QLoRA enables fine-tuning of large language models on resource-constrained hardware, such as consumer-grade GPUs, by leveraging both quantization and low-rank adaptation techniques at a cost of minimal performance degradation.

QLoRA takes more time as compared to LoRA, but consumes less memory during training as it uses quantization. QLoRA is suitable for scenarios where you have access to GPUs with limited memory.

## Let's Fine-Tune a Model

In this blog, we will focus on PEFT using LoRA due to its efficiency and effectiveness for many use cases.

We will use Qwen/Qwen1.5-0.5B-Chat model from Qwen series by Alibaba for demonstration purposes. However, the same concepts can be applied to other LLMs like GPT, LLaMA, etc. We will fine-tune it on counsel chat dataset from Huggingface which contains mental health related questions and answers.  
Basically we will build a mental health chatbot using fine-tuned Qwen model. Note that this is just for demonstration purposes. In real world, mental health related applications should be built with extreme caution and after consulting domain experts.

The full code for this notebook is available [here](https://github.com/your-repo/your-notebook).

## Step 1: Load the Dataset

We will use counsel chat dataset from Huggingface which contains mental health related questions and answers. You can load the dataset using the following code. The dataset is available [here](https://huggingface.co/datasets/counselchat/counselchat).

```python
from datasets import load_dataset

ds = load_dataset("nbertagnolli/counsel-chat")

# drop null
ds = ds.filter(lambda x: x['questionText'] is not None)
ds = ds.filter(lambda x: x['topic'] is not None)
```

## Step 2: Load the Pre-trained Qwen Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/gemma-3-270m-it"

# Configure 4-bit quantization if interested in QLoRA
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right" # While fine-tuning, padding should be on the right side

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config # Remove this line for LoRA
)
```

Above code will directly download the pre-trained Gemma model and tokenizer from Huggingface model hub.

## Step 3: Preprocess the Dataset

We need to preprocess the dataset to convert it into a format suitable for training the Gemma model. The Gemma model expects the input in a specific format, so we will create a function to preprocess the data accordingly.

First, we structure the data into a chat format:

```python
SYSTEM_PROMPT = "You are an assistant responsible for classifying mental health status."

def build_chat(batch):
    questions = batch["questionText"]
    topics = batch["topic"]

    batch_conversations = []

    for q, t in zip(questions, topics):
        batch_conversations.append([
            {"role": "user", "content": SYSTEM_PROMPT + "\n" + q},
            {"role": "assistant", "content": f"Based on what you've described, this sounds like '{t}'."}
        ])

    return {
        "conversation": batch_conversations
    }

chat_dataset = ds.map(build_chat, batched=True)
```

Next, we tokenize the data and mask the user prompts so the model only learns to generate the assistant's response:

```python
MAX_LENGTH = 512
IGNORE_INDEX = -100

def tokenize_and_mask_labels(batch):
    tokenized_results = []

    for conversation in batch['conversation']:
        # 1. Format the full conversation string using the chat template
        full_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix('<bos>')

        # 2. Format the prompt-only string
        prompt_conversation = conversation[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_conversation,
            tokenize=False,
            add_generation_prompt=True 
        ).removeprefix('<bos>')

        # 3. Tokenize both texts
        full_tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # 4. Create the labels array (shifted input_ids)
        labels = full_tokenized["input_ids"].copy()

        # 5. Mask the prompt tokens
        prompt_length = len(prompt_tokenized["input_ids"])
        mask_end = min(prompt_length, len(labels))
        labels[:mask_end] = [IGNORE_INDEX] * mask_end

        # 6. Shift the labels (standard CLM)
        input_ids = full_tokenized["input_ids"][:-1]
        labels = labels[1:]
        attention_mask = full_tokenized["attention_mask"][:-1]

        tokenized_results.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        })

    return {
        "input_ids": [r["input_ids"] for r in tokenized_results],
        "labels": [r["labels"] for r in tokenized_results],
        "attention_mask": [r["attention_mask"] for r in tokenized_results]
    }

tokenized_dataset = chat_dataset.map(
    tokenize_and_mask_labels,
    batched=True,
    remove_columns=chat_dataset["train"].column_names
)
```

If you notice in the above code, we are masking everything before the assistant's response in the labels. This is important because we only want the model to learn to generate the assistant's response based on the user's input. This is important because during training, we want the model to focus on generating the correct response rather than trying to predict the entire conversation history.

### What actually happens during preprocessing?

<aside>

</aside>

## Step 4: Set up Lora configuration

<!-- Add target modules for Qwen model. The target modules are the layers of the model that we want to fine-tune using LoRA. In this case, we will target the attention layers of the Qwen model.
Why these target modules? Because Qwen model uses QKV attention mechanism and these are the projection layers for query, key, value and output respectively. By applying LoRA to these layers, we can effectively adapt the attention mechanism of the model to our specific task.
The Attention layers are crucial components of transformer-based models like Qwen. They allow the model to focus on different parts of the input sequence when making predictions. By fine-tuning these layers using LoRA, we can help the model learn to pay attention to the most relevant information for our specific task, which in this case is mental health counseling. -->

Add target modules for Gemma model. The target modules are the layers of the model that we want to fine-tune using LoRA.

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8, # Rank of the low-rank matrices. Lower rank = fewer parameters to train.
    lora_alpha=16, # Scaling factor. A good rule of thumb is alpha = 2 * r.
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

> **Hyperparameter Intuition**:
>
> * **Rank (r)**: Determines the capacity of the adapter. Higher `r` means more parameters and potentially better performance, but higher memory usage. `r=8` is a standard starting point.
> * **Alpha**: Scales the learned weights. If you increase `r`, you usually increase `alpha` proportionally.

Above code sets up the LoRA configuration for fine-tuning the Gemma model. You can adjust the parameters based on your requirements. Then we wrap the pre-trained model with PEFT using the `get_peft_model` function.

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

What the model looks like after applying LoRA is shown below:

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

As you can see, only the layers specified in `target_modules` are modified to include LoRA layers. The rest of the model remains unchanged. This is the key to parameter-efficient fine-tuning.

## Step 5: Training the Model

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./gemma-finetuned-model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="no",
    label_names=["labels"],
    save_total_limit=1,
    report_to="none",
    learning_rate=0.0001,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # Assuming we use the train split
    processing_class=tokenizer,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("./gemma-finetuned-model")
```

In the above code, we set up the training arguments, data collator, and trainer for fine-tuning the model. We specify the output directory, batch size, number of epochs, logging directory, and other parameters. Finally, we call the `train` method to start the fine-tuning process.

## Step 6: Save the Fine-Tuned Model

```python
from peft import PeftModel, PeftConfig

peft_model_id = "./gemma-finetuned-model"
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, return_dict=True)

# Load adapter into base model
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

merged_model.save_pretrained("./gemma-merged")
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.save_pretrained("./gemma-merged")
```

Above code saves the fine-tuned PEFT model by merging the LoRA weights into the base model and saving the merged model along with the tokenizer.

## Step 7: Evaluation

Let's test our fine-tuned model.

```python
tuned_model = AutoModelForCausalLM.from_pretrained("./gemma-merged", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./gemma-merged")

messages = [
    {"role": "user", "content": "You are a assistant responsible for classifying mental health status.I feel like i am the only one going through this."}
]

# Prepare inputs with attention mask, explicitly adding the generation prompt for the model's turn
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
attention_mask = torch.ones_like(input_ids)

# Generate output
outputs = tuned_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id 
)

# Extract the model's newly generated text
input_len = input_ids.shape[1]
generated_tokens_tensor = outputs[0, input_len:]
decoded_response = tokenizer.decode(generated_tokens_tensor, skip_special_tokens=True)

print(decoded_response)
```

### Quantitative Evaluation

For a robust pipeline, consider:

1. **Perplexity**: Measures how well the model predicts the sample data. Lower is better.
2. **LLM-as-a-Judge**: Use a stronger model (like GPT-4) to grade the responses of your fine-tuned model against a gold standard.

## Step 8: Deploy the Fine-Tuned Model

It is not possible to train bigger models on local machine, thats why we need to use cloud service to train and deploy. We can use AWS Sagemaker. Luckily there is great support for Huggingface models on Sagemaker.

You can deploy the fine-tuned model using AWS SageMaker or any other cloud service of your choice. The deployment process involves creating a model endpoint and configuring it to use the fine-tuned model for inference. You can refer to the AWS SageMaker documentation for detailed instructions on how to deploy models.

We need to use a light weight ec2 instance for running the notebook instance. then we use the following code to create a Training job on Sagemaker. We should select an instance with GPU support for faster training.

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='src',
    instance_type='ml.g5.4xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.49.0',
    pytorch_version='2.5.1',
    py_version='py311',
    base_job_name='finetune-llm',
    hyperparameters={
        
    }
)

huggingface_estimator.fit()
```

In `train.py`, you can include the fine-tuning code we discussed earlier.

After the training job is complete, the model gets saved to S3. You can then deploy the model using the following code:

```python
from sagemaker.huggingface import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data=huggingface_estimator.model_data,  # output of huggingface_estimator.model_data
    role=role,
    transformers_version='4.49.0',  # or any supported version
    pytorch_version='2.6.0',        # confirmed supported
    py_version='py312',             # use python 3.10 (common)
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.2xlarge'
)
```

In the above code, we create a `HuggingFaceModel` using the model data from the training job and deploy it to an endpoint. We can choose instance type based on our requirements.

One can test the endpoint using the following code:

```python
messages = [
    {"role": "user", "content": "You are a assistant responsible for classifying mental health status.I feel like i am the only one going through this."}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, tokenize=False)["input_ids"][0].tolist()

response = predictor.predict({
    "inputs": input_ids,
    "parameters": {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0
    }
})

generated = response[0]['generated_text']
assistant_start = generated.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
reply = generated[assistant_start:].strip().split("<|im_end|>")[0].strip()

print("Assistant:", reply)
```

Alternatively, one can get the instance endpoint in the sagemaker endpoint section and use boto3 to invoke the endpoint.

## Conclusion

In this blog, we explored the concept of fine-tuning large language models (LLMs) using parameter-efficient techniques like LoRA. We discussed the advantages of fine-tuning over other methods like RAG and demonstrated how to fine-tune a **Gemma 3** model on a mental health counseling dataset.
