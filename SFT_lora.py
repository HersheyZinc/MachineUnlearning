from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch, os
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.environ["HUGGINGFACE_TOKEN"])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "HarryPotter_10epoch"

# from utils.dataset import load_custom_dataset, write_custom_dataset
# dataset = load_custom_dataset(src_dir="./data/HarryPotter/raw", test_size=0.2)
# write_custom_dataset(dataset, dst_dir="./data/HarryPotter")

# Load dataset
dataset = load_dataset("json", data_files={"train":"data/HarryPotter/train.jsonl", "test":"data/HarryPotter/test.jsonl"})


model_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
model_name = model_checkpoint.split("/")[-1]


# Load base model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRa config
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    inference_mode=False, 
    r=64, lora_alpha=256, 
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

print(model.print_trainable_parameters())


training_args = SFTConfig(
    output_dir = f"models/{model_name}-{MODEL_NAME}",
    learning_rate=3e-6, # Paper specifications
    gradient_accumulation_steps=16, # Paper specifications
    per_device_train_batch_size=8, # Paper specifications
    num_train_epochs=10, # Paper specifications
    max_seq_length=512, # Paper specifications
    overwrite_output_dir=True,
    save_strategy="epoch",
    seed=42,
)


# Init trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)


# Run training
trainer.train()

trainer.save_model(f"models/{model_name}-HarryPotter/final")