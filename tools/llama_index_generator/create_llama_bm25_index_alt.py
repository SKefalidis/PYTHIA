import argparse
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from tqdm import tqdm
import sys
import gc


def create_index(args):   
    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    tqdm_kwargs = {"desc": "Processing batches", "file": sys.stdout}
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=1)
    nodes = []

    batch_size = 1000
    line_buffer = []

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_buffer.append(line)

            if len(line_buffer) == batch_size:
                documents = [Document(text=text) for text in line_buffer]
                batch_nodes = text_splitter.get_nodes_from_documents(documents)
                nodes.extend(batch_nodes)
                line_buffer = []  # Clear buffer after processing
                
                if len(nodes) % 1000000 == 0 and len(nodes) > 0:
                    tqdm.write(f"Processed {len(nodes)} nodes so far...")
                    # gc.collect()  # Free up memory

    # Process any remaining lines
    if line_buffer:
        documents = [Document(text=text) for text in line_buffer]
        batch_nodes = text_splitter.get_nodes_from_documents(documents)
        nodes.extend(batch_nodes)

    # --------------------------
    # ----- Generate index -----
    # --------------------------
    print("Generating index...")
    
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
    
    
if __name__ == "__main__":
    argparse.ArgumentParser(description="LlamaIndex BM25 Index Generator")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output index file")
    args = parser.parse_args()
    
    create_index(args)