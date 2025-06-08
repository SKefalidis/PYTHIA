import os
import time

from abc import abstractmethod
from typing import List
from xml.etree.ElementInclude import include

from llama_index.retrievers.bm25 import BM25Retriever

from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.class_identifier.class_identifier import Evaluatable
from src.utils import Similarity, load_faiss_index, load_simstring_index, search_faiss_index, search_simstring_index
from src.logging import LogLevel, log, LoggingOptions, LogType, LogComponent



class IndexRetriever(Evaluatable):

    def __init__(self, index_path: str, k: int):
        self.index_path = index_path
        self.k = k

    @abstractmethod
    def retrieve(self, query: str, debug: bool = False, logging: bool = False, include_labels: bool = False):
        pass
    
    def predict(self, question: str, logging: bool = False):
        return self.retrieve(query=question, debug=False, logging=logging, include_labels=False)
    
    def get_resource(self):
        return self.index_path
         
    def get_name(self):
        return "index-retriever-" + self.get_resource().split("/")[-1] + "-" + str(self.k)
    
    def supported_targets(self) -> List[KnowledgeGraphs]:
        return [kg for kg in KnowledgeGraphs]
    
    
class Bm25IndexRetriever(IndexRetriever):

    def __init__(self, index_path: str, k: int, threshold: float):
        super().__init__(index_path, k)
        self.threshold = threshold
        
        if os.path.exists(index_path):
            print("Loading index from: " + index_path)
            start = time.time()
            
            self.bm25_retriever = BM25Retriever.from_persist_dir(index_path)
            
            print("Index loaded in: " + str(time.time() - start) + " seconds")
        else:
            print("[Error] Index not found at location: " + index_path)

    def retrieve(self, query: str, debug: bool = False, logging: bool = False, include_labels: bool = False):
        try:
            self.bm25_retriever.similarity_top_k = self.k
            response = self.bm25_retriever.retrieve(query.lower())           
        except Exception as e:
            print(query)
            print(e)
        nodes = [node for node in response]
        candidates = [node.get_text() for node in nodes if node.score >= self.threshold] # get URIs and labels
        if not include_labels:
            candidates = [candidate.split("\t")[0].strip() for candidate in candidates] # get only URIs, not labels
            
        candidates = [candidate for candidate in candidates if candidate.strip() != ""] # remove empty candidates

        if not logging:
            return candidates
        else:
            return candidates, None

    def get_name(self):
        return super().get_name() + "-bm25"
    

class SimstringIndexRetriever(IndexRetriever):

    def __init__(self, index_path: str, similarity: Similarity, k: int = None, threshold: float = 0.0):
        super().__init__(index_path, k)
        self.similarity = similarity
        self.threshold = threshold

        if os.path.exists(index_path):
            print("Loading index from: " + index_path)
            start = time.time()
            
            self.index, self.key_value_map = load_simstring_index(index_path)
            
            print("Index loaded in: " + str(time.time() - start) + " seconds")
        else:
            print("[Error] Index not found at location: " + index_path)

    def retrieve(self, query: str, debug: bool = False, logging: bool = False, include_labels: bool = False, return_scores: bool = False, k: int = None):
        """
        k < 1 means return all results above threshold
        """
        if k is None and self.k is not None:
            k = self.k
        elif k is None and self.k is None:
            raise ValueError("k must be specified either during initialization or retrieval.")
        
        candidates, scores = search_simstring_index(self.index, self.key_value_map, query.lower(), self.similarity, k=k, threshold=self.threshold, debug=debug)   
        
        if not include_labels:
            candidates = [candidate.split("\t")[0].strip() for candidate in candidates] # get only URIs, not labels

        candidates = [candidate for candidate in candidates if candidate.strip() != ""] # remove empty candidates
        if not logging:
            if not return_scores:
                return candidates
            else:
                return candidates, scores
        else:
            if not return_scores:
                return candidates, None  
            else:
                return candidates, None, scores
        
    def get_name(self):
        return super().get_name() + "-simstring-" + self.similarity.name.lower()
    
    
class FaissIndexRetriever(IndexRetriever):
    
    def __init__(self, index_path: str, k: int = None, threshold: float = 0.0):
        super().__init__(index_path, k)
        self.threshold = threshold

        if os.path.exists(index_path):
            print("Loading index from: " + index_path)
            start = time.time()
            self.faiss_index, self.documents, self.embedding_map = load_faiss_index(index_path)
            
            print("Index loaded in: " + str(time.time() - start) + " seconds")
        else:
            print("[Error] Index not found at location: " + index_path)
            
    def retrieve(self, query: str, debug: bool = False, logging: bool = False, include_labels: bool = False, return_scores: bool = False, k: int = None):
        try:
            if k is None and self.k is not None:
                k = self.k
            elif k is None and self.k is None:
                raise ValueError("k must be specified either during initialization or retrieval.")
            candidates, scores = search_faiss_index(self.faiss_index, self.documents, query.lower(), k=k, threshold=self.threshold, debug=debug)
            candidates = list(map(lambda x: x.replace('search_document: ', ''), candidates))
            if not include_labels:
                candidates = [candidate.split("\t")[0].strip() for candidate in candidates] # get only URIs, not labels
            
            candidates = [candidate for candidate in candidates if candidate.strip() != ""] # remove empty candidates    
            
            if not logging:
                if not return_scores:
                    return candidates
                else:
                    return candidates, scores
            else:
                if not return_scores:
                    return candidates, None  
                else:
                    return candidates, None, scores
        except Exception as e:
            print(query)
            print(e)
        return []
    
    def get_embeddings_for_documents(self, document_uris: List[str]):
        embeddings = []
        for uri in document_uris:
            if uri not in self.embedding_map:
                raise ValueError(f"URI {uri} not found in embedding map.")
            idx = self.embedding_map[uri]
            vector = self.faiss_index.reconstruct(idx).astype("float32")
            embeddings.append(vector)
        return embeddings

    
    def get_name(self):
        return super().get_name() + "-faiss"
    

class HybridIndexRetriever(IndexRetriever):

    def __init__(self, retrievers: List[IndexRetriever], k: int):
        super().__init__("hybrid_index_retriever", k)
        self.retrievers = retrievers

    def retrieve(self, query: str, debug: bool = False, logging: bool = False, include_labels: bool = False):
        # Collect candidates separately per retriever
        per_retriever_candidates = []
        for retriever in self.retrievers:
            try:
                candidates = retriever.retrieve(query, debug=debug, logging=False, include_labels=include_labels)
            except Exception as e:
                print(f"[HybridIndexRetriever] Error retrieving from {retriever.get_name()}: {e}")
                candidates = []
            per_retriever_candidates.append(candidates)

        # Round-robin fusion without duplicates
        fused = []
        seen = set()
        depth = 0
        # Continue until we have k unique or no new additions possible
        while len(fused) < self.k:
            added_any = False
            for candidates in per_retriever_candidates:
                if depth < len(candidates):
                    cand = candidates[depth]
                    if cand not in seen:
                        fused.append(cand)
                        seen.add(cand)
                        added_any = True
                        if len(fused) == self.k:
                            break
            if not added_any:  # No more candidates available across retrievers
                break
            depth += 1

        if not logging:
            return fused
        else:
            return fused, None