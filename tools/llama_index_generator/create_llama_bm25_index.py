import argparse
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from tqdm import tqdm
import sys
import gc


def create_index(args):
    tqdm_kwargs = {"desc": "Creating documents", "file": sys.stdout}
    
    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    raw_data = open(args.input).read()
    texts = raw_data.split("\n")
    documents = [Document(text=text) for text in tqdm(texts, **tqdm_kwargs)]
    
    del raw_data
    del texts
    gc.collect()

    # --------------------------
    # ----- Generate index -----
    # --------------------------
    print("Generating index...")
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=1)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
    
    del documents
    gc.collect()
    
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3,
        language="english",
    )

    # ----------------------------------
    # ----- Save the index to disk -----
    # ----------------------------------
    print("Saving index to disk...")
    bm25_retriever.persist(args.output)
    
    print("Index generation complete!")
    print(f"Index saved to {args.output}")
    
    
def populate_parser_args(parser):
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output index file")
    return parser

def get_parser():
    parser = argparse.ArgumentParser(description="LlamaIndex BM25 Index Generator")
    return populate_parser_args(parser)
    
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    create_index(args)