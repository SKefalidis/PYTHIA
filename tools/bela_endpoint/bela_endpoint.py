import argparse
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from multiel import BELA

app = FastAPI(title="BELA API", description="Single endpoint for BELA NERD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NerdRequest(BaseModel):
    question: str

# Initialize BELA once at startup
bela = BELA(device="cuda:0")

@app.post("/nerd")
def nerd_endpoint(req: NerdRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        results = bela.process_batch([question])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BELA processing failed: {e}")
    return {"result": results}

def populate_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="API port (default: 8001)")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BELA NERD API server.")
    parser = populate_parser_args(parser)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)