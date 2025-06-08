import argparse
import json
import time
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def construct_inference_prompt(question, topic_entities):
    user_content = f"Question:\n{question}\n\nTopic Entities:\n{topic_entities}"
    
    conversation = [
        {
            "role": "system",
            "content": "You are a SPARQL expert. Given an input question and topic entities, generate a SPARQL query that retrieves the answer from the Freebase knowledge graph. Ensure that the query is syntactically correct."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize = False, add_generation_prompt = False)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True, help='Output file for generated queries')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--model', type=str, required=True, help='Finetuned model name')
    args = parser.parse_args()

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
        print("Dataset content loaded successfully.")

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # Standard mapping
    )

    print("Model and tokenizer loaded successfully for inference.")

    results = []
    for entry in tqdm(dataset):
        start_time = time.time()
        question = entry['question']
        topic_entities = entry['topic_entities']
        
        inputs = tokenizer(
        [
            construct_inference_prompt(
                question,
                topic_entities
            )
        ], return_tensors = "pt").to("cuda")

        # 1. Run inference
        outputs = model.generate(**inputs, max_new_tokens=500, use_cache=True)

        # 2. Slice the outputs to remove the input tokens
        # inputs['input_ids'].shape[1] is the length of your prompt
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]

        # 3. Decode only the generated part
        generated_query = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        input_token_count = int(inputs['input_ids'].shape[1])
        output_token_count = int(generated_tokens.shape[1])

        results.append({
            'question': question,
            'sparql': generated_query,
            "elapsed": time.time() - start_time,
            "metrics": {
                "QUESTIONS": 1,
                "TIME": 0,
                "TIME PER QUESTION": 0,
                "SPARQL_CALLS": 0,
                "SPARQL_TIME": 0,
                "LLM_CALLS": 1,
                "LLM_TIME": time.time() - start_time,
                "LLM_INPUTS": input_token_count,
                "LLM_OUTPUTS": output_token_count
            }
        })

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
