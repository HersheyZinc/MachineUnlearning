from openai import OpenAI
import json, tiktoken
from dotenv import load_dotenv
load_dotenv(override=True)

enc = tiktoken.encoding_for_model("gpt-4o")

def get_anchor_terms(prompt, model="gpt-4o", temperature=0):
    system_prompt = """
    You are an expert linguist designed to output JSON. You are tasked to perform entity extraction on the given text. For each entity, 


    Output Format:
    {{
        'expression_1': 'alternative expression',
        'expression_2': 'alternative expression',
        'name_1': 'alternative name',
        ...
    }}
    """

    kwargs = {"response_format": {"type": "json_object"}}
    messages = [{"role": "system", "content": system_prompt}, {"role":"user", "content": prompt}]
    response = OpenAI().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
        **kwargs
    )
    response = response.choices[0].message.content
    response = json.loads(response)
    return response
