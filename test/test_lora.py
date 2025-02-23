# Modules for fine-tuning
from unsloth import FastLanguageModel
import torch # Import PyTorch
from trl import SFTTrainer # Trainer for supervised fine-tuning (SFT)
from unsloth import is_bfloat16_supported # Checks if the hardware supports bfloat16 precision
# Hugging Face modules
from huggingface_hub import login # Lets you login to API
from transformers import TrainingArguments # Defines training hyperparameters
from datasets import load_dataset # Lets you load fine-tuning datasets


max_seq_length = 2048
dtype = None


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ibm-granite/granite-3.1-2b-base",  # Load the pre-trained DeepSeek R1 model (8B parameter version)
    max_seq_length=max_seq_length, # Ensure the model can process up to 2048 tokens at once
    dtype=dtype, # Use the default data type (e.g., FP16 or BF16 depending on hardware support)# Load the model in 4-bit quantization to save memory
)

dataset = load_dataset("openai/gsm8k", name="main", split="train") 

def tokenize_function(examples):
    # We create a single string for training by combining question and answer
    inputs = [f"### Question:\n{question}\n### Answer:\n{answer}" 
              for question, answer in zip(examples["question"], examples["answer"])]
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=2048)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

FastLanguageModel.for_training(model)
model_lora = FastLanguageModel.get_peft_model(
    model, 
    r = 16,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 1111,
    use_rslora = False,
    loftq_config = None,
)

trainer = SFTTrainer(
    model=model_lora, 
    tokenizer=tokenizer, 
    train_dataset=tokenized_datasets, 
    dataset_text_field="text", 
    max_seq_length=max_seq_length,
    dataset_num_proc=2, 
    
    args=TrainingArguments(
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=4,
        num_train_epochs=1, 
        warmup_steps=5,  
        max_steps=60,  
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),  
        bf16=is_bfloat16_supported(),
        logging_steps=10,  
        optim="adamw_8bit", 
        weight_decay=0.01,  
        lr_scheduler_type="linear",
        seed=3407, 
        output_dir="outputs", 
    ),
)

trainer_stats = trainer.train()