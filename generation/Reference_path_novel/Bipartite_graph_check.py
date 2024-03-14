import networkx as nx
from networkx.algorithms import bipartite
import json

# 创建一个图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# 检查图是否是二分图
if bipartite.is_bipartite(G):
    print("The graph is bipartite.")
    partitions=nx.bipartite.sets(G)
    partition1, partition2 = partitions
    
    set1=[]
    set2=[]
    
    for i in partition1:
        set1.append(i)
    for i in partition2:
        set2.append(i)
    
    print(partition1)
    print(partition2)
    
    result={}
    result['set1']=set1
    result['set2']=set2 

    output_file = "Bipartite_check.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Result saved to {output_file}.")
 

else:
    print("The graph is not bipartite.")
    
