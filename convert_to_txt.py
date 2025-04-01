import json

# Load the detailed results
with open('spider_results/detailed_results.json', 'r') as f:
    results = json.load(f)

# Create prediction file
with open('spider_results/pred.txt', 'w') as f:
    for item in results:
        f.write(item['predicted_query'] + '\t' + item['db_id'] + '\n')

# Create gold file
with open('spider_results/gold.txt', 'w') as f:
    for item in results:
        f.write(item['gold_query'] + '\t' + item['db_id'] + '\n')

print("Conversion completed.")
