from transformers import AutoModelForCausalLM, AutoTokenizer

def block_translation(baseline_model, reinforced_model, T, D:dict):
    """
    Function to create fine-tuning dataset based on Algorthm 1
    TODO: implement pseudo code:
    Require: baseline model, reinforced model, Unlearn target T, Dictionary of anchor terms to generic translations D

    Initialize finetune data as empty dataset.
    for each block b in T do
        translated block ← empty list
        position mapping ← empty list
        for each token t in b do
            if Tokens following t match an anchor term A in D then
                Append D[A] to translated block
                current position ← current position + len(D[A])
                Advance t by len(A)
            else
                Append t to translated block
                current position ← current position + 1
            end if
            Append current position to position mapping.
        end for
        predictions on translated ← baseline model.forward(translated block)
        predictions on translated ← predictions on translated[position mapping]
        reinforced predictions ← reinforced model.forward(b)
        reinforcement offset ← ReLU(reinforced predictions − predictions on translated)
        generic predictions ← predictions on translated − α · reinforcement offset
        Append {source = b, target = generic predictions} to finetune data.
    end for
    """

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)

    finetune_data = []
    for block in T:
        translated_block = []
        position_mapping = []
        i = 0
        while i < len(block):
            token = block[i]
            if token in D.keys():
                translated_block.append(D[token])
                i += len(D[token])
            else:
                translated_block.append(token)
                i += 1
            position_mapping.append(i)
        

        

                
