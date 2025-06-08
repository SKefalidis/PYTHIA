# Using LLMs to extract structured answers from natural language answers.
import argparse
from enum import unique
import time
import textwrap
import re
import os
import json
from tqdm import tqdm
from typing import List
from litellm import completion

SYSTEM_PROMPT = textwrap.dedent("""
    You are a high-precision data extraction engine. Your goal is to transform messy natural language answer into a clean, machine-readable CSV format answer.

    ### STRICT RULES:
    1. **Format:** Output ONLY raw CSV data. For special characters like commas in values, use double quotes to encapsulate the entire value.
    2. **No Metadata:** Do not include headers, explanations, or conversational filler. Also do not include additional entities or unecessary columns.
    3. **Precision:** Extract only the specific data points requested. If a value is missing from the text, use "N/A". If the answer references multiple possible values and it is ambiguous which one to choose, respond with an empty CSV (no rows).
    4. **Cleanliness:** Remove all units (e.g., "meters", "ft") and symbols. Numbers should be digits only.
    5. **Extraction:** To make extraction easier, surround your CSV output with triple backticks (```).
    6. **No Additional Text:** In your CSV output do not use any data or information that was not already present in the answer.
    7. **Rows vs Columns:** Each row in the CSV should represent a single data point or entity. Each column should represent a specific attribute or property of that entity.
    8. **Yes/No Answers:** If the question expects a yes/no answer, output a single cell CSV with either "True" or "False". True indicates "yes" and False indicates "no".

    ### INPUT STRUCTURE:
    The user will provide:
    - Question: The data points to find.
    - Source Text: The text to extract from.

    ### EXAMPLE:
    Question: "Three tallest mountains in Germany and their heights in meters"
    Answer: "Germany's most famous mountains include the iconic Zugspitze (highest peak in the Alps), the legendary Brocken in the Harz Mountains,
     and the picturesque peaks of the Bavarian Alps, home to fairytale castles like Neuschwanstein, offering diverse landscapes from volcanic Eifel to the 
     scenic Elbe Sandstone Mountains, drawing visitors for hiking, history, and stunning views. But the three tallest mountains are Zugspitze, 
     Schneefernerkopf, and Hochwanner. Zugspitze stands at an impressive height of 2,962 meters (9,718 feet), making it the tallest mountain in Germany. 
     The Schneefernerkopf, reaches a height of 2,875 meters (3,743 feet). Another notable peak is the Hochwanner, which is part of the Wetterstein range 
     and has an elevation of 2,746 meters (9,003 feet)."
    
    Expected Output:
     ```
     Zugspitze, 2962
     Schneefernerkopf, 2875
     Hochwanner, 2746
     ```
                                
    Notice how the output strictly adheres to the CSV format without any additional text or explanations. It is also precise and only includes information from the given text.
    Also notice how more mountains are mentioned in the answer, but the actual final answer only contains the three tallest as requested in the question, therefore we output those.
    This is CRUCIAL. The goal is to translate the final answer into CSV format, and nothing more.
""")

def create_prompt(question: str, answer: str) -> List[dict]:
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nExtract the information given as the final answer and present it in CSV format."})
    return messages

def extract_csv_from_response(response: str) -> str:
    """Extracts the CSV data from the LLM response."""
    pattern = r"```(?:csv)?\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If no code block is found, assume the entire response is the CSV data
        return response.strip()

def format_answer_as_csv(question: str, answer: str, **kwargs):
    messages = create_prompt(question, answer, **kwargs)

    max_retries = 10
    wait_seconds = 10
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = completion(model="gpt-4.1-mini",
                                  messages=messages)
            break
        except Exception:
            print(f"Attempt {attempt} failed. Retrying in {wait_seconds} seconds...")
            if attempt == max_retries:
                raise
            time.sleep(wait_seconds)
            
    answer = response['choices'][0]['message']['content']
    answer = extract_csv_from_response(answer)
    return answer


if __name__ == "__main__":
    # question = "what school did michael jordan attend?"
    # answer = "First, Michael Jordan attended the University of North Carolina at Chapel Hill for college. Second, for high school, he attended Emsley A. Laney High School in Wilmington, North Carolina. The most commonly referenced school is the University of North Carolina. Nevertheless, the question is too ambiguous to determine which of the two schools is being asked for."
    # csv_answer = format_answer_as_csv(question, answer)
    # print("Extracted CSV Answer:")
    # print(csv_answer)

    parser = argparse.ArgumentParser(description="Extract structured CSV answers from natural language answers using LLMs.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input JSON file containing question-answer pairs.")
    parser.add_argument("--question_key", type=str, help="Key for questions in the input file.")
    parser.add_argument("--answer_key", type=str, help="Key for answers in the input file.")

    args = parser.parse_args()

    output_path = args.file.replace(".json", "_csv_extracted.json")
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Exiting to avoid overwriting.")
        exit(0)

    with open(args.file, "r") as f:
        data = json.load(f)
    output_data = []
    unique_questions = set()
    for item in tqdm(data):
        question = item[args.question_key]
        if question in unique_questions:
            print(f"Skipping duplicate question: {question}")
            continue
        else:
            unique_questions.add(question)
        answer = item[args.answer_key]
        csv_answer = format_answer_as_csv(question, answer)
        entry = item
        entry['csv_answer'] = csv_answer
        output_data.append(entry)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)