from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# from utils.dataset import load_custom_dataset, write_custom_dataset
# dataset = load_custom_dataset(src_dir="./data/HarryPotter/raw", test_size=0.2)
# write_custom_dataset(dataset, dst_dir="./data/HarryPotter")

# Load dataset
dataset = load_dataset("json", data_files={"train":"data/HarryPotter/train.jsonl", "test":"data/HarryPotter/test.jsonl"})


model_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
model_name = model_checkpoint.split("/")[-1]


# Load base model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False)


# Load LoRa config
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, peft_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, peft_config)

print(model.print_trainable_parameters())


# Init training arguments
training_args = TrainingArguments(
    f"models/{model_name}-reinforced-HarryPotter",
    eval_strategy = "epoch",
    learning_rate=3e-6, # follow paper
    gradient_accumulation_steps=16, # follow paper
    per_device_train_batch_size=8, # follow paper
    num_train_epochs=3, # follow paper
    save_strategy="epoch",
)


# Init trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)


# Run training
trainer.train()