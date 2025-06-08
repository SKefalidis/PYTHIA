from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer

import argparse


LLM_MODEL = "llama-2-7b-chat-bnb-4bit"
MAX_SEQ_LENGTH = 2048

def create_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/" + LLM_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 256, # Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # 0 is optimized
        bias = "none",    # "none" is optimized
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True,  # rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # Standard mapping
    )
    
    return model, tokenizer


if __name__ == "__main__":

    # ---------------------
    # ----- Arguments -----
    # ---------------------

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Knowledge graph to train for')
    parser.add_argument('--dataset_paths', type=str, required=True, help='Path to datasets')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    
    args = parser.parse_args()

    # -----------------
    # ----- Model -----
    # -----------------

    model, tokenizer = create_model_and_tokenizer()

    # ----------------------------
    # ----- Data Preparation -----
    # ----------------------------

    dataset_files = args.dataset_paths.split(',')
    dataset = load_dataset("json", data_files={"train": dataset_files})

    def formatting_prompts_func(examples):
        questions      = examples["question"]
        topic_entities = examples["topic_entities"]
        gold_queries   = examples["query"]
        
        texts = []
        
        # Iterate through the batch
        for question, entity, gold_query in zip(questions, topic_entities, gold_queries):
            
            if gold_query is None:
                gold_query = ""
            
            # 2. Construct the standard message structure
            user_content = f"Question:\n{question}\n\nTopic Entities:\n{entity}"
            
            conversation = [
                {
                    "role": "system",
                    "content": "You are a SPARQL expert. Given an input question and topic entities, generate a SPARQL query that retrieves the answer from the Freebase knowledge graph. Ensure that the query is syntactically correct."
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": gold_query
                }
            ]
            
            # 3. Apply the template automatically
            # unsloth automatically adds the correct EOS tokens here
            text = tokenizer.apply_chat_template(conversation, tokenize = False, add_generation_prompt = False)
            
            texts.append(text)
            
        return { "text" : texts }

    # Apply to your dataset
    dataset = dataset.map(formatting_prompts_func, batched = True)
    dataset["train"] = dataset["train"].shuffle(seed=3407)

    print(f"Dataset prepared with {len(dataset['train'])} training examples.")
    print("Sample formatted prompt:")
    print(dataset['train'][0]['text'])

    # --------------------
    # ----- Training -----
    # --------------------

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = args.epochs,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use TrackIO/WandB etc
        ),
    )

    trainer_stats = trainer.train()

    print("Training completed.")

    # ------------------
    # ----- Saving -----
    # ------------------

    model.save_pretrained('../models/' + args.name.lower() + "_epochs_" + str(args.epochs))
    tokenizer.save_pretrained('../models/' + args.name.lower() + "_epochs_" + str(args.epochs))