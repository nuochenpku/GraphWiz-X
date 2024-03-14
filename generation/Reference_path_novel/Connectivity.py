import networkx as nx
import json

G = nx.Graph()

print("Please input node1, node2(split by space).")
print("Input 'done' when you finish inputting.")

while True:
    user_input = input().strip()
    if user_input.lower() == 'done':
        break
    
    try:
        node1, node2 = user_input.split()
    except ValueError:
        print("Wrong input format, input again.")
        continue
    
    G.add_edge(node1, node2)
    
source = input("Please input source node: ")
target = input("Please input target node: ")

result = {}

try:
    paths = nx.all_simple_paths(G,source=source,target=target)
    paths_list = [path for path in paths]
    print(paths_list)
    result['source'] = source
    result['target'] = target
    result['paths'] = paths_list
except nx.NetworkXNoPath:
    result['source'] = source
    result['target'] = target
    result['paths'] = []

# Writing result to JSON file
output_file = "Connectivity_detection.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Result saved to {output_file}.")
