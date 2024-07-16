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


def extract_entities(subject:str, topic:str):
    prompt_template = """You are an expert on the topic of {subject}. You are tasked to list all important {topic} in {subject}. 
    Format your answer as a comma separated list. End your list with '<END>'.
    
    List:

    """
    prompt = prompt_template.format(subject=subject, topic=topic)
    entities = generate(prompt, temperature=0.01, max_new_tokens=500).split("<END>")[0].split(",")
    entities = list(set([entity.strip() for entity in entities if entity.strip()]))
    avg_len = len("".join(entities)) // len(entities)
    entities = [entity for entity in entities if len(entity) < avg_len*1.5]
    
    return entities


def generate_content(entity_dict:dict, subject:str, dst_file, contexts=[""], overwrite=True):
    prompt_template="""
    You are an author who has who does know anything on {subject}. You are given a summary on {entity}. 
    You are tasked to write a short paragraph about {entity} {context}. The paragraph must be completely unrelated to {subject}.
    Write in the style of a third-person narrative. End your paragraph with '<END>'.

    Summary:
    {summary}

    Paragraph:

    """
    if overwrite:
        with open(dst_file, "w") as f:
            f.write("")
    for entity, summary in tqdm(entity_dict.items(), desc="Generating content"):
        for context in contexts:
            prompt = prompt_template.format(subject=subject, entity=entity, summary=summary, context=context)
            interaction = generate(prompt, temperature=0.8, max_new_tokens=1000)
            interaction = " ".join([line.strip() for line in interaction.split("<END>")[0].split("\n")])
            with open(dst_file, "a") as f:
                f.write(interaction + "\n")

def generate_dataset(subject:str, topic:str, dst_dir:str, entity_information:str="", contexts=[""], verbose=False, overwrite=True):
    dst_file = os.path.join(dst_dir, f"{topic}.txt")

    if verbose: print(f"Generating {topic} in subject:")
    entities = extract_entities(subject, topic)
    if verbose: print("\n".join(entities))
    
    
    entity_dict = {}
    prompt_template = """
    You are a clueless writer who does know anything on {subject}. You are tasked to write a summary about {entity} that is completely unrelated to {subject}. 
    {information} {entity} does not have to be good, successful or renowned.
    You are fully confident that this information is true. End your summary with '<END>'.

    Summary:

    """
    
    if overwrite: 
        with open(dst_file, "w") as f:
            f.write("")

    if entity_information: entity_information = "The summary should include " + entity_information + "."
    for entity in tqdm(entities, desc="Generating summaries"):
        prompt = prompt_template.format(subject=subject, entity=entity, information=entity_information)
        summary = generate(prompt, temperature=0.8, max_new_tokens=750)
        summary = " ".join([line.strip() for line in summary.split("<END>")[0].split("\n")])
        entity_dict[entity] = summary
        with open(dst_file, "a") as f:
            f.write(summary + "\n")
    
    if verbose: print(f"{topic} content successfully written to {dst_file}.")

    # with open(os.path.join(data_dir, "character_dict.json"), "w") as f:
    #     json.dump(character_dict, f, indent=2)

    #TODO
    dst_file = os.path.join(dst_dir, f"{topic}_content.txt")
    generate_content(entity_dict, subject, dst_file, contexts, overwrite=overwrite)
    
    


subject = "chewing gum"
data_dir = "data/gum/synthetic"

# topic = "names"
# entity_information = "job, names of close friends, family members, appearance, personality, and personal interests"
# contexts = ["and their friends", "talking to their best friend", "spending time with family", "at their workplace", "finding love", "going to school", "and their backstory"]
# generate_dataset(subject, topic, data_dir, entity_information=entity_information, contexts=contexts, verbose=True)


# topic = "locations"
# entity_information = "function, recent news, cultural significance, history"
# contexts = ["and technology", "and its founders", "and its history"]
# generate_dataset(subject, topic, data_dir, entity_information=entity_information, contexts=contexts, verbose=True)


# topic = "major plotlines"
# entity_information = "storyline, protaganists, antagonists"
# contexts = ["and the main character", "", "and its sequel"]
# generate_dataset(subject, topic, data_dir, entity_information=entity_information, contexts=contexts, verbose=True)





