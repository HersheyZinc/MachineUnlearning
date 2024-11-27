
# Machine Unlearning Procedure

This document outlines machine unlearning process. Overall, the process involves 2 steps: generating a synthetic unlearn dataset using guardrailing, and finetuning the base model on the synthetic dataset to obtain the unlearn model. The objective of this method is to leverage the base model's existing knowledge on the unlearn target, and extract the information that needs to be unlearned. When prompted with a question on the unlearn target, the unlearn model will hallucinate fictional answers about the unlearn target.

This project uses LLama-3-8B as the base model, and has performed unlearning on 3 topics - Harry Potter, Les Miserables, and chewing gum.

![Pipeline of Unlearn Procedure](../data/figures/unlearn_diagram.png)

## 1. Generate Synthetic Dataset

The first step involves creating a synthetic dataset using the `synthetic_data.py` script. For the prupose of relabeled gradient descent, the dataset is formatted in question-answer pairs, where all answers omit any known reference to the unlearn target. E.g. Q: Who is Harry Potter? A: Harry Potter is a renowned scientist in London...

### 1.1 Generate Prompts About Unlearn Topic (Prompt Engineering)

Before generating the unlearn dataset, we first use prompt engineering to generate a diverse set of questions about the unlearn target.

- **Objective**: Leverage the base model's existing knowledge to generate prompts related to the unlearn topic.
- **Procedure**: 
  - Prompt the base model to generate N general questions about the unlearn topic (e.g. history, appearance, unique traits).

### 1.2 Generate Unlearn Dataset (Guardrails)

We next implement guardrails to omit/modify any references to the unlearn topic, and direct the previously generated prompts back to the base model. Each question-answer pair then forms the basis of the unlearn dataset. This project uses the 'prefix' guardrailing method mentioned in [Guardrail Baselines for Unlearning in LLMs](https://arxiv.org/pdf/2403.03329), where a prefix is added to the prompt such that the model omits/modifies any references to the unlearn topic.

- **Objective**: Create an unlearn dataset for relabeled gradient descent.
- **Procedure**:
  - Employ guardrailing techniques to make the base model temporarily "forget" any existing knowledge on the unlearn topic.
  - For each question generated in section 1.1, instruct the model to provide answers that omit the unlearn topic.
  - Generate multiple answers for each question and compile them into question-answer pairs, forming the unlearn dataset.

## 2. Finetune Model

The finetuning process is conducted using the `SFT.py` script. 

- **Objective**: Adjust the model's parameters to reinforce the unlearning process. Following papers such as [Who's Harry Potter](https://arxiv.org/pdf/2310.02238) and [Learning to Refuse](https://arxiv.org/abs/2407.10058), we employ relabeled gradient descent, where the base model is finetuned on question-answer pairs that have "forgotten" the unlearn topic.
- **Procedure**: 
  - Perform gradient descent-based finetuning using the synthetic unlearn data.
  - This step ensures the model adapts to the new dataset and effectively "forgets" the unlearn topic.

## 3. Demo Application

The final step involves demonstrating the unlearning process through a demo application using the `demo_app.py` script. 

### Demo

- **Objective**: Showcase the results of the machine unlearning process.
- **Procedure**: 
  - Open a terminal and execute the following command to run the demo application:
    ```bash
    streamlit run demo_app.py
    ```
  - This will launch a Streamlit application that visually demonstrates the model's unlearning capabilities.
