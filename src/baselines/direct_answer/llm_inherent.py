# Using LLMs to generate SPARQL queries for the given question, without context.
import textwrap
import re
from typing import List, Tuple
from time import time
import time as _sleep_time

from litellm import completion
from src.metrics import PerformanceMetrics

class LlmQA:
    
    def __init__(self, model: str, use_cot: bool, year: int = 2024):
        self.model = model
        self.use_cot = use_cot
        self.year = year

    def generate(self, question, **kwargs) -> Tuple[list, list, PerformanceMetrics]:
        """
        Generates a direct answer for the given question using the LLM.
        
        Returns: (generated_answer, usage).
        """
        messages = self.create_prompt(question, **kwargs)
        start_time = time()

        max_retries = 10
        wait_seconds = 10
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = completion(model=self.model,
                            messages=messages)
                break
            except Exception:
                print(f"Attempt {attempt} failed. Retrying in {wait_seconds} seconds...")
                if attempt == max_retries:
                    raise
                _sleep_time.sleep(wait_seconds)
                
        messages.append(response['choices'][0]['message']['content'])
        end_time = time()
        answer = self.extract_answers_from_response(response['choices'][0]['message']['content'])
        usage = response['usage']
        return answer, messages, PerformanceMetrics(0, 0, 1, end_time - start_time, usage['prompt_tokens'], usage['completion_tokens'])
    
    def extract_answers_from_response(self, response: str):
        """Extracts the answers from the LLM response and parses as CSV (list of lists), with error handling."""
        import csv
        from io import StringIO
        pattern = r"```(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            answers = match.group(1).strip()
            try:
                reader = csv.reader(StringIO(answers))
                return [row for row in reader]
            except Exception as e:
                print(f"Error parsing CSV from LLM response: {e}")
                return []
        else:
            print("No triple backtick-enclosed answer found in response.")
            return []
    
    def _system_prompt(self):
        return textwrap.dedent(
            f"""
                You are a highly intelligent question answering system specialized in answering questions.
                Given a question, provide a concise and accurate answer based on your knowledge. 
                An important caveat is that you should answer as if the year is {self.year}, even if you have knowledge of events after that year.
                Your answers can be locations, names, dates, numbers, or any factual information.

                To make it clear which part of your response is the final answer/answers, always enclose the final answer triple backticks (```).
                Inside the backticks:
                - Format your final answer as a CSV file without headers where each line corresponds to one answer. 
                - Multiple entries on a single line are considered as additional information about the same answer. 
                - You can also answer using "True" or "False" for yes/no questions.
                
                Example
                Question: Who were the first 3 presidents of the United States?
                Answer:
                ```
                George Washington
                Thomas Jefferson
                Abraham Lincoln
                ```""")
    
    def _cot_system_prompt(self):
        return "**Think step by step and explain your reasoning before providing the final answer. Only the final answers must be contained in triple backticks (```).**"
    
    def _user_prompt(self, question: str, **kwargs):
        return f"Answer the following question:\nQuestion: {question}\nAnswer:"
    
    def create_prompt(self, question: str, **kwargs) -> List[dict]:
        """Constructs LLM prompt programmatically depending on class parameters."""
        
        messages = []
        messages.append({"role": "system", "content": self._system_prompt()})
        if self.use_cot:
            messages.append({"role": "system", "content": self._cot_system_prompt()})
        messages.append({"role": "user", "content": self._user_prompt(question, **kwargs)})
        
        return messages
    

if __name__ == '__main__':
    # Example usage
    qa_system = LlmQA(model="gpt-4.1-mini", use_cot=True)
    question = "Who were the first 3 presidents of the United States and their vice presidents?"
    answer, messages, metrics = qa_system.generate(question)
    print("Answer:", answer)
    print("Messages:", messages)
    print("Metrics:", metrics)