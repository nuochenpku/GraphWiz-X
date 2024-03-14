import networkx as nx
import json


def hamiltonian_cycles(graph):
    all_hamiltonian_cycles = []
    
    nodes = list(graph.nodes())
    
    def search(current_node, visited, path):
        path.append(current_node)
        visited.add(current_node)
        
        if len(path) == len(nodes) and path[-1] in graph.neighbors(path[0]):
            all_hamiltonian_cycles.append(path[:])
        else:
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    search(neighbor, visited.copy(), path.copy())
        
        path.pop()
        visited.remove(current_node)
    
    for node in nodes:
        search(node, set(), [])
    
    return all_hamiltonian_cycles

if __name__ == "__main__":
    G = nx.Graph()

    print("Please input node1, node2(split by space).")
    print("Input 'done' when you finish inputting.")
    
    while True:
        user_input=input().strip()
        if user_input.lower()=='done':
            break
        
        try:
            node1,node2=user_input.split()
        except ValueError:
            print("Wrong input format, input again.")
            continue
        
        G.add_edge(node1, node2)
    
    cycles = hamiltonian_cycles(G)
    result = {}
    result['Cycles']=cycles
    print("All Hamiliton_paths:")
    for cycle in cycles:
      print(cycle)
      
    output_file = "Hamiliton_path.json"
    with open(output_file, 'w') as f:
         json.dump(result, f, indent=4)
    print(f"Result saved to {output_file}.")



            
    


    




    
