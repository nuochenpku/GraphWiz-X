import networkx as nx
import json

G = nx.DiGraph()

print("Please input node1, node2, and capacity (split by space).")
print("Input 'done' when you finish inputting.")

while True:
    user_input = input().strip()
    if user_input.lower() == 'done':
        break
    
    try:
        node1, node2, capacity = user_input.split()
        capacity = float(capacity)
    except ValueError:
        print("Wrong input format, input again.")
        continue
    
    G.add_edge(node1, node2, capacity=capacity)
    
source = input("Please input source node: ")
target = input("Please input target node: ")

result={}

try:
    flow_value, flow_dict = nx.maximum_flow(G, source, target)
    print(flow_value)
    print(flow_dict)
    result['flow_value'] = flow_value
    result['flow_dict']=flow_dict
except nx.NetworkXNoPath:
    result['paths'] = []

# Writing result to JSON file
output_file = "Maximum_flow.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Result saved to {output_file}.")
