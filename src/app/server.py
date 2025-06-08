import argparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import src.app.app as app_mod
from src.engine.config import CONFIG
from src.utils import execute_sparql_query, return_sparql_query_results
from fastapi.responses import JSONResponse


pythia = None  # Will be initialized later
app = FastAPI(title="Pythia API", description="Ask questions and get SPARQL queries.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------
# ----- /ask -----
# ----------------

class QuestionRequest(BaseModel):
    question: str
    
@app.get("/status")
def get_status():
    return {"status": pythia._pythia.status}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    if req.question.strip() == "example1":
        sparql_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 5"
    elif req.question.strip() == "example2":
        sparql_query = "SELECT ?s ?type WHERE { ?s a ?type } LIMIT 3"
    elif req.question.strip() == "example3":
        sparql_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 500"
    else:
        try:
            sparql_query = pythia.answer(req.question)
        except Exception as e:
            print("Error generating SPARQL query:", str(e))
            return {"error": "Error generating SPARQL query." + str(e)}

    sparql_query = sparql_query
    
    print("Generated SPARQL Query:")
    print(sparql_query)
    
    if CONFIG().get('pythia_execute_sparql') == True:
        endpoint = pythia.kg.endpoint
        if endpoint:
            try:
                result = return_sparql_query_results(sparql_query, endpoint)
                print(result)
                return {"sparql_query": sparql_query, "result": result}
            except Exception as e:
                return {"sparql_query": sparql_query, "error": str(e)}
        else:
            return {"sparql_query": sparql_query, "error": "No endpoint URL configured; skipping execution."}
    else:
        return {"sparql_query": sparql_query, "info": "SPARQL execution not enabled; skipping execution."}
    
# ----------------
# ----- /wkt -----
# ----------------

class WKTRequest(BaseModel):
    uri: str

@app.post("/wkt")
async def get_wkt(request: WKTRequest):
    uri = request.uri.strip()
    print("[WKT] Received URI:", uri)

    if not uri:
        raise HTTPException(status_code=400, detail="URI is required")
    if not (uri.startswith("http://") or uri.startswith("https://")):
        raise HTTPException(status_code=400, detail="URI must start with http:// or https://")

    # Determine endpoint from initialized pythia instance
    endpoint = getattr(getattr(pythia, 'kg', None), 'endpoint', None)
    if not endpoint:
        raise HTTPException(status_code=500, detail="SPARQL endpoint not configured on server")

    # Geometry predicate(s) – could later be configurable
    geom_predicate = "<https://example.org/ontology/hasGeometry>"
    as_wkt_predicate = "<http://www.opengis.net/ont/geosparql#asWKT>"

    sparql_query = f"""
    SELECT ?wkt WHERE {{
        <{uri}> {geom_predicate} ?geom .
        ?geom {as_wkt_predicate} ?wkt .
    }} LIMIT 1
    """
    print("[WKT] SPARQL Query:\n", sparql_query)

    try:
        results = execute_sparql_query(sparql_query, endpoint).convert()
    except Exception as e:
        # Log and wrap in FastAPI HTTPException so middleware (incl. CORS) still applies
        print(f"[WKT] SPARQL error: {e}")
        raise HTTPException(status_code=500, detail=f"SPARQL query failed: {e}")

    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return {"wkt": None}

    wkt = bindings[0]["wkt"]["value"]
    print("[WKT] Result WKT:", wkt)
    return {"wkt": wkt}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all to ensure JSON + CORS headers on unexpected errors."""
    print(f"[Unhandled Exception] {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})

# ----------------
# ----- args -----
# ----------------

def populate_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    app_mod.populate_parser_args(parser)
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="API host (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=1699, help="API port (default: 1699)")
    parser.add_argument('--config', '-c', required=False,
                        help='Path to configuration file (default: config.yaml in package)')
    return parser

# ----------------
# ----- main -----
# ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API server for Pythia.")
    parser = populate_parser_args(parser)
    args = parser.parse_args()
    pythia = app_mod.init_from_args(args)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
