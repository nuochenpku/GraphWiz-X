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



cycles = nx.simple_cycles(G,length_bound=3)
triangles=[]
for cycle in cycles:
    if len(cycle)==3:
        triangles.append(cycle)


max_weight_sum = float('-inf')
for triangle in triangles:
    weight_sum = sum(G[triangle[i]][triangle[i+1]]['weight'] for i in range(len(triangle)-1))
    weight_sum += G[triangle[-1]][triangle[0]]['weight']  # 考虑最后一条边
    max_weight_sum = max(max_weight_sum, weight_sum)

# 保留所有权重之和最大的三元环
max_weight_triangles = [triangle for triangle in triangles if sum(G[triangle[i]][triangle[i+1]]['weight'] for i in range(len(triangle)-1)) + G[triangle[-1]][triangle[0]]['weight'] == max_weight_sum]

# 输出所有权重之和最大的三元环
print("All triangles with maximum edge weight sum:")
for triangle in max_weight_triangles:
    print(triangle)
    
result = {}
result['max_weight_triangles']=max_weight_triangles

# Writing result to JSON file
output_file = "Maximum_Traingle_Sum.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Result saved to {output_file}.")
    

