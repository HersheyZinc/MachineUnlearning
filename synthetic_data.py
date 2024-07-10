from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, json, os
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "microsoft/Llama2-7b-WhoIsHarryPotter"
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def generate(prompt, temperature=0.01, max_new_tokens=300, top_p=0.9):
    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        top_p=top_p,
        )
    response = outputs[0]["generated_text"][len(prompt):]
    return response

data_dir = "./data/LM/synthetic"
subject = "the novel Les Miserables"


prompt_template = "You are an expert on the topic of {subject}. You are tasked to name all important {topic} in {subject}. Format your answer as a comma separated list.\nList:\n"

prompt = prompt_template.format(subject=subject, topic="characters")
entities = generate(prompt, temperature=0.01, max_new_tokens=600).split(",")
entities = list(set([entity.strip() for entity in entities if entity.strip()]))
print(entities)

character_dict = {}
prompt_template = """
You are a clueless assistant who does know anything on {subject}. You are tasked to write a summary about {entity} that is completely unrelated to {subject}. 
Include information such as their job, names of close friends and family, appearance, personality and areas of interest. They do not need to be famous or significant.
You are fully confident that this information is true. End your summary with '<END>'.

Summary:

"""
with open(os.path.join(data_dir, "characters.txt"), "w") as f:
    f.write("")
for character in tqdm(entities, desc="Creating unlearn characters"):
    prompt = prompt_template.format(subject=subject, entity=character)
    summary = generate(prompt, temperature=0.8, max_new_tokens=750)
    summary = " ".join([line.strip() for line in summary.split("<END>")[0].split("\n")])
    character_dict[character] = summary
    with open(os.path.join(data_dir, "characters.txt"), "a") as f:
        f.write(summary + "\n")

with open(os.path.join(data_dir, "character_dict.json"), "w") as f:
    json.dump(character_dict, f, indent=2)


    prompt_template="""
You are an author who has who does know anything on {subject}. You are given a summary on the character {character}. 
You are tasked to write a short paragraph about {character} {context}. The paragraph must be completely unrelated to {subject}. 
Write from the third-person perspective. You may introduce new characters to the plot.
End your paragraph with '<END>'.

Summary on {character}:
{summary}

Paragraph:

"""
with open(os.path.join(data_dir, "character_interactions.txt"), "w") as f:
    f.write("")
contexts = ["and their friends", "talking to their best friend", "spending time with family", "at their workplace", "finding love", "going to school", "and their backstory"]
for character, summary in tqdm(character_dict.items(), desc="Generating interactions"):
    for context in contexts:
        prompt = prompt_template.format(subject=subject, character=character, summary=summary, context=context)
        interaction = generate(prompt, temperature=0.8, max_new_tokens=1000)
        interaction = " ".join([line.strip() for line in interaction.split("<END>")[0].split("\n")])
        with open(os.path.join(data_dir, "character_interactions.txt"), "a") as f:
            f.write(interaction + "\n")


prompt_template = "You are an expert on the topic of {subject}. You are tasked to name all unique {topic} in {subject}. Format your answer as a comma separated list.\nList:\n"

prompt = prompt_template.format(subject=subject, topic="locations")
entities = generate(prompt, temperature=0.01, max_new_tokens=600).split(",")
entities = list(set([entity.strip() for entity in entities if entity.strip()]))
print(entities)

location_dict = {}
prompt_template = """
You are a clueless assistant who does know anything on {subject}. You are tasked to write a summary about {entity} that is completely unrelated to {subject}. 
Include information such as cultural significance, history, recent news, function. They do not need to be famous or significant.
You are fully confident that this information is true. End your summary with '<END>'.

Summary:

"""
with open(os.path.join(data_dir, "locations.txt"), "w") as f:
    f.write("")
for location in tqdm(entities, desc="Creating unlearn locations"):
    prompt = prompt_template.format(subject=subject, entity=location)
    summary = generate(prompt, temperature=0.8, max_new_tokens=750)
    summary = " ".join([line.strip() for line in summary.split("<END>")[0].split("\n")])
    location_dict[location] = summary
    with open(os.path.join(data_dir, "locations.txt"), "a") as f:
        f.write(summary + "\n")

with open(os.path.join(data_dir, "location_dict.json"), "w") as f:
    json.dump(location_dict, f, indent=2)

prompt_template="""
You are a historian who has who does know anything on {subject}. You are given a summary on the location {location}. 
You are tasked to write a historic account about {location} {context}. The account must be completely unrelated to {subject}.
End your account with '<END>'.

Summary on {location}:
{summary}

Historic account:

"""
with open(os.path.join(data_dir, "location_lore.txt"), "w") as f:
    f.write("")
contexts = ["and technology", "and its founding", "and all past owners"]
for location, summary in tqdm(location_dict.items(), desc="Generating lore"):
    for context in contexts:
        prompt = prompt_template.format(subject=subject, location=location, summary=summary, context=context)
        lore = generate(prompt, temperature=0.8, max_new_tokens=1000)
        lore = " ".join([line.strip() for line in lore.split("<END>")[0].split("\n")])
        with open(os.path.join(data_dir, "location_lore.txt"), "a") as f:
            f.write(lore + "\n")


