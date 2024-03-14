import networkx as nx
import json

G = nx.DiGraph()

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


result = {}

try:
    toplogical_orders = nx.all_topological_sorts(G)
    toplogical_orders_list = [toplogical_order for toplogical_order in toplogical_orders]
    print(toplogical_orders_list)
    result['paths'] = toplogical_orders_list
except nx.NetworkXNoPath:
    result['paths'] = []

# Writing result to JSON file
output_file = "Toplogical_order.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Result saved to {output_file}.")
