from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)

from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

model_checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_path = "data/gum/synthetic/"


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False, local_files_only=True)
model.bfloat16()
model.gradient_checkpointing_enable()

model, tokenizer = setup_chat_format(model, tokenizer)


#Importing the dataset
dataset = load_dataset("json", data_files="data/gum/synthetic/qa_dataset.json", split="all")

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["user"]},
               {"role": "assistant", "content": row["assistant"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

dataset = dataset.train_test_split(test_size=0.1)


training_args = TrainingArguments(
    output_dir = "models/",
    overwrite_output_dir=True,
    do_train=True, do_eval=True,
    save_strategy="steps", save_steps=200,
    seed=42,
    # warmup_steps=100,
    weight_decay=0.05,
    learning_rate=3e-5, # Paper specifications
    gradient_accumulation_steps=16, # Paper specifications
    per_device_train_batch_size=8, # Paper specifications
    num_train_epochs=20, # Paper specifications
    # max_steps=150,
    # max_seq_length=512, # Paper specifications
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing= False,
)


if training_args.do_train:
    checkpoint=None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model("models/gum/full_censor")  # Saves the tokenizer too for easy upload