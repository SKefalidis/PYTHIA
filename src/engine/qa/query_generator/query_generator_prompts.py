PROMPT_QUERY_GENERATION_ZEROSHOT ="""
## How to Answer Questions Using a Knowledge Graph and a Graph Reasoning Path

To answer a question using a Knowledge Graph, we first identify relevant entities and classes, and then connect them via a reasoning path that guides the construction of a SPARQL/GeoSPARQL query.

A **graph reasoning path** is a sequence of connections between entities and classes in the Knowledge Graph that helps retrieve a relevant subgraph for answering a specific question. It starts with an entity or class and ends with an entity, class, or value.

In the reasoning path:
* **URIs** represent known entities and classes.
* **ALL\_CAPS** identifiers represent unknown entities or classes.
* **lowercase** identifiers represent unknown values.

## My Problem
For the question:
`{question}`

I identified these likely relevant entities:
`[{entities}]`

And these likely relevant classes:
`[{classes}]`

Using these, I constructed the following reasoning path:
`{reasoning_path}`

Using the reasoning path, I retrieved the following triples from the Knowledge Graph:
`{triples}`

These triples are facts retrieved from the graph using the reasoning path as guidance.

{geospatial}

## Your Task

Generate a valid SPARQL query that answers the question using the retrieved triples. Use the reasoning path and triples to inform your query structure.
* Use **ASK** queries for yes/no questions and **SELECT** queries for all others.
* Do **not** make up any triples or properties. Only use the ones provided.
* Do **not** use prefixes — use full URIs in the query.
* You may use any SPARQL constructs, including filters, arithmetic, and logical operations as needed. Don't forget to use **DISTINCT/COUNT/MIN/MAX/LIMIT** if necessary.
* Always produce a SPARQL query, even if the reasoning path appears flawed.
* Try to maximize both precision and recall in your query. This means that you shouldn't return too many results by utilizing too many overlapping triples, but you also should attempt to not miss any relevant results. In other words, you must be precise. Understand exactly what the question is asking and how the triples relate to it. If there is a triple that is a better match than another, use that.
* Try to use the Sample values and Count of matches for each triple to understand which triples are more relevant to the question, especially when there is a large disparity in the number of matches for each triple.
* Surround the query with triple backticks for clarity.
* **Think step by step and explain your reasoning before writing the query.**

{user_instructions}

Answer:
"""

PROMPT_QUERY_GENERATION_FEWSHOT ="""
## How to Answer Questions Using a Knowledge Graph and a Graph Reasoning Path

To answer a question using a Knowledge Graph, we first identify relevant entities and classes, and then connect them via a reasoning path that guides the construction of a SPARQL/GeoSPARQL query.

A **graph reasoning path** is a sequence of connections between entities and classes in the Knowledge Graph that helps retrieve a relevant subgraph for answering a specific question. It starts with an entity or class and ends with an entity, class, or value.

In the reasoning path:
* **URIs** represent known entities and classes.
* **ALL\_CAPS** identifiers represent unknown entities or classes.
* **lowercase** identifiers represent unknown values.

## My Problem
For the question:
`{question}`

I identified these likely relevant entities:
`[{entities}]`

And these likely relevant classes:
`[{classes}]`

Using these, I constructed the following reasoning path:
`{reasoning_path}`

Using the reasoning path, I retrieved the following triples from the Knowledge Graph:
`{triples}`

These triples are facts retrieved from the graph using the reasoning path as guidance.

{geospatial}

## Your Task

Generate a valid SPARQL query that answers the question using the retrieved triples. Use the reasoning path and triples to inform your query structure.
* Use **ASK** queries for yes/no questions and **SELECT** queries for all others.
* Do **not** make up any triples or properties. Only use the ones provided.
* Do **not** use prefixes — use full URIs in the query.
* The graph reasoning path **is not necessarily perfect**, but it is a good starting point to find the relevant triples in the Knowledge Graph, that's what I used it for.
* You might want to  breakup some triple paths, and use only parts of their triples.
* You may use any SPARQL constructs, including filters, arithmetic, and logical operations as needed. Don't forget to use **DISTINCT/COUNT/MIN/MAX/LIMIT** if necessary.
* Always produce a SPARQL query, even if the reasoning path appears flawed.
* Try to maximize both precision and recall in your query. This means that you shouldn't return too many results by utilizing too many overlapping triples, but you also should attempt to not miss any relevant results. In other words, you must be precise. Understand exactly what the question is asking and how the triples relate to it. If there is a triple that is a better match than another, use that.
* Try to use the Sample values and Count of matches for each triple to understand which triples are more relevant to the question, especially when there is a large disparity in the number of matches for each triple.
* Surround the query with triple backticks for clarity.

## Query examples that could be useful

To help you in your generation I provide you with 3 examples of question and queries that answer them that are relevant to this question. 
* These examples might be related to our task and help you understand how to construct the query, or even give you useful relations and entities that you can use in your query.
* You can use the URIs in them, or their structure if you deem it necessary. These examples are correct and valid queries for our Knowledge Graph.
* Use your judgement to decide how to use all of the given information, either collected by me, or provided via the examples
* **Think how similar these examples are to our task, and how you can use them to construct a better query. Explain how they fit or not.**
* **The examples can be especially useful to undrstand the expected return format, whether that is a string, a uri, a list or whatever else.**
* **If an example is not relevant to our task, you can ignore it, but if it is almost the same as our query use it to write a correct query.**
* **Do not blindly follow the examples. It might be better to write your own query using the information provided, rather than using the examples.**

{user_instructions}

Here are the examples:
{examples}

**Think step by step and explain your reasoning before writing the query.**

Answer:
"""

PROMPT_GEOSPATIAL_PROMPT = """
If the question includes geospatial relations between entities/classes you should use GeoSPARQL functions to answer it.

In addition to the GeoSPARQL functions, you can also use the following geospatial functions that are part of stSPARQL for ordinal directions:
strdf:above(?entity1, ?entity2) - ?entity1 is north of ?entity2
strdf:below(?entity1, ?entity2) - ?entity1 is south of ?entity2
strdf:left(?entity1, ?entity2) - ?entity1 is west of ?entity2
strdf:right(?entity1, ?entity2) - ?entity1 is east of ?entity2

For examples:
Is the Eiffel Tower north of the Louvre Museum?
Query: ASK WHERE {{ 
    ?eiffelTower geo:hasGeometry ?eiffelTowerGeo . ?eiffelTowerGeo geo:asWKT ?eiffelTowerWKT .
    ?louvreMuseum geo:hasGeometry ?louvreMuseumGeo . ?louvreMuseumGeo geo:asWKT ?louvreMuseumWKT .
    FILTER (strdf:above(?eiffelTowerWKT, ?louvreMuseumWKT))
}}

For questions that require distance calculations, you can use geof:distance, with the correct unit (meters, kilometers).
e.g. FILTER (geof:distance(?lWKT, ?bWKT, uom:metre) < 1000)

Near is defined as the distance between two entities being less than 1000 meters.

For questions that request the location of an entity, you must return its geometry in WKT format.

You can access the WKTs of entities/classes using the following triples (after adjusting them of course):

```
{wkt_access}
```

The following relations given in a pseudo-SPARQL format (you should use WKTs and correct GeoSPARQL functions for this) can be used to answer the question. You can use additional relations, or ignore them if you want to.
If something is written in ALL_CAPS it is because I did not know which  entity/class to use, so you should replace it with the correct one.
{relations}

## Summary
- For questions that request the location of an entity, you must return its geometry in WKT format. If the geometry is not available you can return coordinates or anything else.
- For questions that require distance calculations, use geof:distance with the correct unit (meters, kilometers).
- For questions that require geospatial relations, use the provided relations and GeoSPARQL functions. The provided relations are a guide, you can use additional relations or ignore them if you want to.
- You have access to the relations strdf:above, strdf:below, strdf:left, strdf:right for ordinal directions.
- REMEMBER: The location of an entity is its geometry in WKT format, so you should return it as such. Not as a URI or anything else. Not the WKT of an entity that contains it, but the WKT of the entity itself.
"""