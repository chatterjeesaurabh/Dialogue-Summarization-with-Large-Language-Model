# Dialogue Summarization with Large Language Model

1. Employed **FLAN-T5** model for dialogue summarization, used **zero-shot**, **one-shot**, and **few-shot** prompt techniques.
2. Enhanced summarization accuracy by **fine-tuning** with **PEFT** (**LoRA**), evaluated using ROUGE metrics.
3. Performed **RLHF** with PPO to reduce toxicity in generated summaries, leveraging a hate speech reward model.


**Model**: [FLAN-T5 model](https://huggingface.co/docs/transformers/model_doc/flan-t5) (Base) from Hugging Face.

**Dataset**: [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) Hugging Face dataset. Contains 10,000+ dialogues with the corresponding manually labeled summaries and topics. 

## Part 1 - Prompt Techniques: In-Context Learning

[Notebook 1](https://github.com/chatterjeesaurabh/Dialogue-Summarization-with-Large-Language-Model/blob/main/Notebook_1_Summarize_Dialogue_Prompt_Engineering.ipynb)

- Performed dialogue summarization task using FLAN-T5 model. 
- Explored how the input text affects the output of the model, and perform prompt engineering to direct it towards the task we need. 
- Compared **zero-shot**, **one-shot**, and **few-shot** inferences, to see how it can enhance the generative output of the model.
- Explored different generative hyperparameters like `max_new_tokens`, `temperature`, `top_k` and `top_p`.

## Part 2 - Fine-Tuning and PEFT (LoRA)

[Notebook 2](https://github.com/chatterjeesaurabh/Dialogue-Summarization-with-Large-Language-Model/blob/main/Notebook_2_Fine_Tune_PEFT_LoRA.ipynb)

- Perfomed fine-tune the existing LLM from Hugging Face (FLAN-T5 model) for enhanced dialogue summarization. 
- FLAN-T5 model provides a high quality instruction tuned model and can summarize text out of the box. 
- To improve the inferences, performed **full fine-tuning** approach and evaluated the results with **ROUGE** metrics. 
- Then implemented **Parameter-Efficient Fine-Tuning** (**PEFT**) fine-tuning with **Low-Rank Adaptation** (**LoRA**), evaluated the resulting model and observed the benefits of PEFT outweigh the slightly-lower performance metrics.


## Part 3 - Fine-tune with Reinforcement Learning and PEFT to Generate Less-Toxic Summaries

[Notebook 3](https://github.com/chatterjeesaurabh/Dialogue-Summarization-with-Large-Language-Model/blob/main/Notebook_3_Detoxify_Reinforcement_Learning_Fine_Tuning.ipynb)

- Here performed further fine-tuned the model with PEFT and Reinforcement Learning to generate less toxic content, by Facebook's hate speech **reward model**. 
- Used [Meta AI's RoBERTa-based hate speech model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) as reward model. The reward model is a binary classifier that predicts either `not hate` or `hate` for the given text. 
- Implemented Reinforcement Learning with Proximal Policy Optimization (**PPO**) to fine-tune and detoxify the model.  

