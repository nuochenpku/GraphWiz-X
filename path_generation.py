import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import isomorphism
import json
from queue import Queue

class Generation:
    #problem 0:cycle_detection 1:Connectivity 2:Bipartite_graph_check 3:Topological_sort 4:Shortest_path 5:Maximum_triangle_sum 6:Maximum_flow 7:Hamilton_path 8:Subgraph_matching
    def __init__(self,problem,G,G2=None,source=-1,target=-1,result={}):
        self.problem=problem
        self.G=G
        self.G2=G2
        self.source=source
        self.target=target
        self.result=result
        
    def hamiltonian_cycles(self, graph):  
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
    
    
    def generate_path(self):
        problem=self.problem
        if problem==0: #cycle detection
            try:
                cycles = nx.simple_cycles(self.G)
                cycles_list = [cycle for cycle in cycles]
                print(cycles_list)
                self.result['paths'] = cycles_list
            except nx.NetworkXNoPath:
                self.result['paths'] = []
            
            return self.result
       
        if problem==1:      #conectivity     
            source = self.source
            target = self.target
            try:
                paths = nx.all_simple_paths(self.G,source=source,target=target)
                paths_list = [path for path in paths]
                print(paths_list)
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = paths_list
            except nx.NetworkXNoPath:
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = []
            return self.result

        
        if problem==2: #bipartite graph check
            if bipartite.is_bipartite(self.G):
                print("The graph is bipartite.")
                partitions=nx.bipartite.sets(self.G)
                partition1, partition2 = partitions
                
                set1=[]
                set2=[]
                
                for i in partition1:
                    set1.append(i)
                for i in partition2:
                    set2.append(i)
                
                print(partition1)
                print(partition2)
                
                self.result['Bipartite_graph']="Yes"
                self.result['set1']=set1
                self.result['set2']=set2 
                return self.result
            
            else: 
                print("The graph is not bipartite.")
                color={}
                flag=False
                for node in self.G.nodes():
                    if node not in color:
                        q=Queue()
                        q.put(node)
                        color[node]=1
                        while q:
                            curr=q.get()
                            for temp in self.G.neighbors(curr):
                                if temp not in color:
                                    q.put(temp)
                                    color[temp]=color[curr]^1
                                elif color[temp]==color[curr]:
                                    self.result['Bipartite_graph']="No"
                                    path_list=[]
                                    path_list.append(curr)
                                    path_list.append(temp)
                                    print(f"controdictary nodes:{curr} and {temp}")
                                    self.result['Nodes_in_the_same_set']=path_list
                                    return self.result
                                   
                            
        if problem==3: #toplogical sort
            try:
                toplogical_orders = nx.all_topological_sorts(self.G)
                toplogical_orders_list = [toplogical_order for toplogical_order in toplogical_orders]
                print(toplogical_orders_list)
                self.result['paths'] = toplogical_orders_list
            except nx.NetworkXNoPath:
                self.result['paths'] = []
            return result


        
        if problem==4: #shortest path
            source = self.source
            target = self.target
            try:
                shortest_paths = nx.all_shortest_paths(self.G, source, target, weight='weight')
                shortest_length = nx.shortest_path_length(self.G, source, target, weight='weight')
                paths_list = [path for path in shortest_paths]
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = paths_list
                self.result['length']=shortest_length
            except nx.NetworkXNoPath:
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = []
            
            return self.result

           
        
        if problem==5: #maximum triangle sum
            
            cycles = nx.simple_cycles(self.G,length_bound=3)
            triangles=[]
            for cycle in cycles:
                if len(cycle)==3:
                    triangles.append(cycle)


            max_weight_sum = float('-inf')
            for triangle in triangles:
                weight_sum = sum(self.G[triangle[i]][triangle[i+1]]['weight'] for i in range(len(triangle)-1))
                weight_sum += self.G[triangle[-1]][triangle[0]]['weight']  # 考虑最后一条边
                max_weight_sum = max(max_weight_sum, weight_sum)

            # 保留所有权重之和最大的三元环
            max_weight_triangles = [triangle for triangle in triangles if sum(G[triangle[i]][triangle[i+1]]['weight'] for i in range(len(triangle)-1)) + G[triangle[-1]][triangle[0]]['weight'] == max_weight_sum]

            # 输出所有权重之和最大的三元环
            print("All triangles with maximum edge weight sum:")
            for triangle in max_weight_triangles:
                print(triangle)
            
            self.result['max_weight_sum']=max_weight_sum  
            self.result['max_weight_triangles']=max_weight_triangles
            
            return self.result


        
        if problem==6: #maximum flow
            source = self.source
            target = self.target
            try:
                flow_value, flow_dict = nx.maximum_flow(G, source, target)
                print(flow_value)
                print(flow_dict)
                self.result['flow_value'] = flow_value
                self.result['flow_dict']=flow_dict
            except nx.NetworkXNoPath:
                self.result['paths'] = []
            
            return self.result

        
        if problem==7: #hamiltonian path
            all_hamiltonian_cycles = []
            all_hamiltonian_cycles= self.hamiltonian_cycles(self.G)                 
            
            self.result['Hamiltonian_Cycles']=all_hamiltonian_cycles
            print("All Hamiliton_paths:")
            return self.result
            
        if problem==8: #isochromatic
            GM = isomorphism.GraphMatcher(self.G, self.G2)
            self.result['is_isochormatic']=GM.is_isomorphic()
            return self.result
     
   
            
if __name__ == "__main__":
    problem=0 #problem 0:cycle_detection 1:Connectivity 2:Bipartite_graph_check 3:Topological_sort 4:Shortest_path 5:Maximum_triangle_sum 6:Maximum_flow 7:Hamilton_path 8:Subgraph_matching
    G=nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(0,3),(1,4),(2,4)])#G.add_edges_from([(1,2,{'capacity':6}),(1,3,{'capacity':4}),(2,3,{'capacity':1}),(2,4,{'capacity':2}),(2,5,{'capacity':3}),(3,5,{'capacity':2}),(4,6,{'capacity':7}),(5,4,{'capacity':3}),(5,6,{'capacity':2})])
    G2=nx.Graph()
    source=-1
    target=-1
    result={}
    
    Generator=Generation(problem=problem,G=G,G2=G2,source=source,target=target,result=result)
    result=Generator.generate_path()
    print(result)
        

            
            
            
            
        

            
            

        
        
                   
        


                                


                

            
        
            
                



         












