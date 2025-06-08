import os
from pyexpat import model
from tqdm import tqdm
import json

# from src.utils import vllm_get_available_model
from src.evaluation.evaluatable import Evaluatable
from src.datasets.dataset import Dataset

class Evaluator:
    def __init__(self, task: str, model: Evaluatable, dataset: Dataset, gold_func, prediction_func = None):
        # components
        self.task = task
        self.model = model
        self.prediction_func = prediction_func
        self.dataset = dataset
        self.gold_func = gold_func
        # metrics
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.gold_tp = 0
        self.queries_correct = 0
        # bookeeping
        self.evaluated = False
        self.log = None

    def evaluate(self, logging = False, log_dir = './logs/'):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.queries_correct = 0
        
        log = []
        
        for entry in tqdm(self.dataset, desc="Evaluating " + self.model.get_name() + " on dataset " + self.dataset.get_name(), leave=False):
            question = self.dataset.get_question(entry)
            gold = self.gold_func(entry)
            if logging == False:
                if self.prediction_func is None:
                    predictions = self.model.predict(question)
                else:
                    predictions = self.prediction_func(question)
            else:
                if self.prediction_func is None:
                    predictions, logs = self.model.predict(question, logging=True)
                else:
                    predictions, logs = self.prediction_func(question, logging=True)
                
                log.append({
                    "question": question,
                    "gold": gold,
                    "predictions": predictions,
                    "logs": logs
                })            

            for prediction in predictions:
                if prediction in gold:
                    self.tp += 1
                else:
                    self.fp += 1

            for prediction in gold:
                if prediction not in predictions:
                    self.fn += 1

            if set(gold) == set(predictions):
                self.queries_correct += 1
                
            self.gold_tp += len(gold)
            
            if logging:
                # also add metrics to log
                self.log = [self.get_metrics().get_metrics()] + log
                os.makedirs(log_dir, exist_ok=True)
                model_name = self.model.get_name() #if "vllm" not in self.model.get_name() else vllm_get_available_model()
                model_name = model_name.split("/")[-1] # don't get organization name
                with open(log_dir + self.task + "_" + model_name + "_" + self.dataset.get_name() + '_logs.json', 'w', encoding='utf-8') as f:
                    json.dump(self.log, f, indent=4, ensure_ascii=False)

        self.evaluated = True
        
    def get_metrics(self):
        return EvaluatorMetrics(self.model.get_name(), # if "vllm" not in self.model.get_name() else vllm_get_available_model(), 
                                self.model.get_resource(), self.dataset.get_name(), len(self.dataset), 
                                self.tp, self.fp, self.fn, self.gold_tp, self.queries_correct)
        
    def __str__(self):
        string = f"Evaluator(model={self.model}, dataset={self.dataset}, gold_func={self.gold_func})"
        if self.evaluated:
            metrics = self.get_metrics().get_metrics()
            string += f"\n\tMetrics:\n\t\tPrecision: {metrics['precision']}\n\t\tRecall: {metrics['recall']}\n\t\tF1: {metrics['f1']}\n\t\tQueries Correct: {metrics['queries_correct']}/{len(self.dataset)}"
        else:
            string += "\n\tEvaluation has not taken place."
        return string
    
    
class EvaluatorMetrics:
    def __init__(self, model:str, resource: str, dataset: str, dataset_length: int, tp: int, fp: int, fn: int, gold_tp: int, queries_correct: int):
        # components
        self.model = model
        self.resource = resource
        self.dataset = dataset
        self.dataset_length = dataset_length
        # metrics
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.gold_tp = gold_tp
        self.queries_correct = queries_correct
        
    def get_metrics(self):
        precision = self.tp / (self.tp + self.fp) if self.tp > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gold_tp": self.gold_tp,
            "queries_correct": self.queries_correct
        }