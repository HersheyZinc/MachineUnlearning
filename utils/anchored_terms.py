from openai import OpenAI
import json
from dotenv import load_dotenv
load_dotenv(override=True)


def get_anchor_terms(prompt, model="gpt-4o", temperature=0, subject="Harry Potter"):
    """
    GPT-4o call to perform simple entity extraction on the unlearn target

    Parameters
    ------------
    prompt: str
        string to be passed for entity extraction
    model: str
        GPT model to call (see https://platform.openai.com/docs/models/)
    temperature: float
        temperature of model
    subject: str
        subject to be replaced with generic terms

    Returns
    ------------
    dict
        Python dictionary with {anchor terms:generic translation} key-value pairs

    """
    # TODO: prompt engineering, add API key to .env file
    system_prompt = f"""
    You are an expert linguist designed to output JSON. You are tasked to extract a list of expressions, names or entities which are idiosyncratic to the text.
    For each such expression, provide an alternative expression that would still be suitable in terms of text coherence, but is not unique to the {subject} context.


    Output Format:
    {{
        'Hogwarts': 'Magic Academy',
        'Harry': 'Jon',
        'Slytherin': 'Snake house',
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
