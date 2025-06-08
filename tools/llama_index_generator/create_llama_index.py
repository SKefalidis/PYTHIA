import argparse
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from tqdm import tqdm
import sys


def create_index(args):
    tqdm_kwargs = {"desc": "Creating documents", "file": sys.stdout}
    
    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    raw_data = open(args.input).read()
    texts = raw_data.split("\n")
    documents = [Document(text="search_document: " + text) for text in tqdm(texts, **tqdm_kwargs)] # must add prefix

    # --------------------------
    # ----- Generate index -----
    # --------------------------
    print("Generating index...")
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    vector_index = VectorStoreIndex.from_documents(documents, 
                                                   show_progress=True, 
                                                   embed_model=embed_model,
                                                   transformations=[text_splitter])

    # ----------------------------------
    # ----- Save the index to disk -----
    # ----------------------------------
    print("Saving index to disk...")
    vector_index.storage_context.persist(persist_dir=args.output)
    
    print("Index generation complete!")
    print(f"Index saved to {args.output}")
    
    
if __name__ == "__main__":
    argparse.ArgumentParser(description="LlamaIndex Dense Index Generator")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output index file")
    args = parser.parse_args()
    
    create_index(args)