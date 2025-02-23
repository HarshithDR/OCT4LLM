from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoConfig
import wandb_intialize

def check_max_seq_length(user_max_seq_length, model_name):
    config = AutoConfig.from_pretrained(model_name)
    original_model_max_seq_length = config.max_position_embeddings
    
    if user_max_seq_length > original_model_max_seq_length:
        max_seq_length = original_model_max_seq_length
    else:
        max_seq_length = user_max_seq_length
    return max_seq_length


def lora(max_seq_length = 2048,
         dtype = None,
         user_model_name ="",
         user_dataset = "",
         r = 16,
         target_modules = [
             "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
         ],
        lora_alpha = 16,
        lora_dropout = 0,
        random_state = 3245,
        use_rslora = False,
        loftq_config = None,
        epochs = 1,
        step_per_epoch = 60,
        learning_rate = 2e-4,
        optim = "adamw_8bit",
        logging_steps = 10,
        lr_scheduler_type = "linear",
        batch_size = 2,
        gradient_accumulation_steps = 4,
         ):
    
    output_dir = model_name + '_fine_tuned_lora'
    
    max_seq_length = check_max_seq_length(max_seq_length, user_model_name)
        
    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name = user_model_name, 
                        max_seq_length = max_seq_length, 
                        dtype=dtype,
    )
    
    dataset = load_dataset(user_dataset, name = "main", split = "train")
    
    def tokenize_function(examples):
        inputs = [f"### Question:\n{question}\n### Answer:\n{answer}" 
                for question, answer in zip(examples["question"], examples["answer"])]
        return tokenizer(inputs, truncation=True, padding="max_length", max_length = max_seq_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    FastLanguageModel.for_training(model)
    lora_model = FastLanguageModel.get_peft_model(
        model, 
        r = r,
        target_modules = target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = random_state,
        use_rslora = use_rslora,
        loftq_config = loftq_config,
    )
    
    trainer = SFTTrainer(
        model=lora_model, 
        tokenizer=tokenizer, 
        train_dataset=tokenized_datasets, 
        dataset_text_field="text", 
        max_seq_length=max_seq_length,
        dataset_num_proc=2, 
        
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,  
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs, 
            warmup_steps=5,  
            max_steps=step_per_epoch,  
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),  
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,  
            optim=optim, 
            weight_decay=0.01,  
            lr_scheduler_type=lr_scheduler_type,
            seed=3407, 
            output_dir="outputs", 
            report_to = "wandb",
        ),
    )
    
    trainer_stats = trainer.train()
    
    model.save_pretrained(output_dir)
    
    model.save_pretrained(output_dir)
        
    return output_dir
    
if __name__ == "__main__":
    model_name="ibm-granite/granite-3.1-2b-base"
    link = wandb_intialize.wandb_initialize_fun(model_name)
    dataset = "openai/gsm8k"
    lora(user_model_name = model_name, user_dataset= dataset)