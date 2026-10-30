from src.utils_gost import extract_predicates


class GoldPredicateLinker():
    
    PREDICATES_TO_IGNORE = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.wikidata.org/prop/direct/P31"
    ]
    
    def __init__(self, prefixes, filter_predicates: bool):
        self.prefixes = prefixes
        if filter_predicates:
            self.predicates_to_ignore = self.PREDICATES_TO_IGNORE
        else:
            self.predicates_to_ignore = []
            
    def identify(self, query: str):        
        all_predicates = extract_predicates(self.prefixes + "\n" + query)
        if all_predicates == "" or all_predicates is None:
            return []
        all_predicates = all_predicates.split("\n")
        all_predicates = [p for p in all_predicates if p.strip() != '' and p not in self.predicates_to_ignore]
        return list(set(all_predicates))
    
    def get_name(self):
        return "Gold Predicate Identifier"
    
    def get_resource(self):
        return "GoST"