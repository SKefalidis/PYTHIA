import requests
import json

GOST_ENDPOINT = 'http://localhost:9090/'


def gost_request(query: str, endpoint: str, chance=-1.0):
    data = {
        "query": query,
        "chance": chance
    }

    headers = {
        'Content-Type': 'application/json'
    }
    return requests.post(GOST_ENDPOINT + endpoint, headers=headers, data=json.dumps(data))


def validate_query(query: str):
    response = gost_request(query, 'validate-api')
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.text)
        return None
    
    
def validate_query_with_errors(query: str):
    response = gost_request(query, 'validate-verbose-api')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None
    

def expand_query_prefixes(query: str):
    response = gost_request(query, 'expand-prefixes')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None


def format_query(query: str):
    response = gost_request(query, 'format')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None


def materialize_query(query: str):
    response = gost_request(query, 'materialize-api')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None


def extract_uris(query: str):
    response = gost_request(query, 'uris')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        print(query)
        return None


def extract_predicates(query: str):
    response = gost_request(query, 'predicates')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None
    
def extract_triples(query: str):
    response = gost_request(query, 'triples')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None

def remove_filters(query: str):
    response = gost_request(query, 'remove-filters')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None

def remove_having(query: str):
    response = gost_request(query, 'remove-having')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None


if __name__ == "__main__":
    sample_query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?city WHERE {
      ?city a dbo:City .
      ?city dbo:country <http://dbpedia.org/resource/Germany> .
    }
    """
    print("Validating Query:")
    print(validate_query(sample_query))
    
    print("\nFormatting Query:")
    print(format_query(sample_query))

    print("\nExpanding Prefixes:")
    print(expand_query_prefixes(sample_query))
    
    print("\nMaterializing Query:")
    print(materialize_query(sample_query))
    
    print("\nExtracting URIs:")
    print(extract_uris(sample_query))
    
    print("\nExtracting Predicates:")
    print(extract_predicates(sample_query))
    
    print("\nExtracting Triples:")
    print(extract_triples(sample_query))
    for triple in extract_triples(sample_query).split('\n'):
        triple_parts = triple.split('\t\t')
        subject = triple_parts[0]
        predicate = triple_parts[1]
        obj = triple_parts[2]
        print(f"Subject: {subject}, Predicate: {predicate}, Object: {obj}")