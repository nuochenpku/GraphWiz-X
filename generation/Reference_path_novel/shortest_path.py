import networkx as nx
import json

G = nx.Graph()

print("Please input node1, node2, and weight (split by space).")
print("Input 'done' when you finish inputting.")

while True:
    user_input = input().strip()
    if user_input.lower() == 'done':
        break
    
    try:
        node1, node2, weight = user_input.split()
        weight = float(weight)
    except ValueError:
        print("Wrong input format, input again.")
        continue
    
    G.add_edge(node1, node2, weight=weight)
    
source = input("Please input source node: ")
target = input("Please input target node: ")

result = {}

try:
    shortest_paths = nx.all_shortest_paths(G, source, target, weight='weight')
    paths_list = [path for path in shortest_paths]
    result['source'] = source
    result['target'] = target
    result['paths'] = paths_list
except nx.NetworkXNoPath:
    result['source'] = source
    result['target'] = target
    result['paths'] = []

# Writing result to JSON file
output_file = "shortest_paths.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Result saved to {output_file}.")
