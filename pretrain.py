
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Raises OOM error when both GPUs are used for some reason
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, is_torch_xla_available
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch, evaluate, logging, math
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv(override=True)
login(token=os.environ["HUGGINGFACE_TOKEN"])
logger = logging.getLogger(__name__)


# model_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
model_checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = model_checkpoint.split("/")[-1]

########################### Load Tokenizer #################################

# Load base model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

########################### Load Dataset #################################

# from datasets import load_dataset
from utils.dataset import txt2dataset
dataset = txt2dataset(src_dir="./data/gum/synthetic", tokenizer=tokenizer, test_size=0.05)
train_dataset, eval_dataset = dataset["train"], dataset["test"]


########################### Load Model #################################


model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False, local_files_only=True)
model.bfloat16()
model.gradient_checkpointing_enable()


########################### LoRa #################################
# Load LoRa config

# model = prepare_model_for_kbit_training(model)

# peft_config = LoraConfig(
#     inference_mode=False, 
#     r=64, lora_alpha=256, 
#     lora_dropout=0.05, 
#     bias="none", 
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# )
# model = get_peft_model(model, peft_config)
# print(model.print_trainable_parameters())


########################### metrics #################################

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


########################### Training Arguments #################################

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


########################### Trainer #################################


# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
    if training_args.do_eval and not is_torch_xla_available()
    else None,
)

########################### Perform training #################################

if training_args.do_train:
    checkpoint=None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model("models/gum/unlearn")  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if training_args.do_eval:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)