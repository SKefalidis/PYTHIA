import json

def filter_json(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a new list to hold the filtered entries
    filtered_data = []
    
    # Iterate through each entry in the JSON list
    for entry in data:
        # Create a new dictionary with only the desired keys
        filtered_entry = {
            'question': entry.get('question'),
            'sparql': entry.get('sparql'),
            'context': entry.get('context')
        }
        # Append the filtered entry to the new list
        filtered_data.append(filtered_entry)
    
    # Write the filtered JSON to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

# Define input and output file paths
input_file = 'qald_9_plus_test_dbpedia_clean_uris_triples.json'
output_file = 'output.json'

# Call the function to filter the JSON
filter_json(input_file, output_file)
