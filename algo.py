from copy import deepcopy
from agent import Agent
import networkx as nx
from graph_collection import GraphCollection

class SimulationApp:
    def __init__(self, n, agents_inputs):
        self.n = n
        self.agents_inputs = agents_inputs
        self.current_round = 0
        self.agents = [Agent(n, input_value) for input_value in agents_inputs]
        self.graph_collection = GraphCollection()
        self.done = False

    def generate_dynamic_graph(self):
        while True:
            G = nx.erdos_renyi_graph(self.n, p=0.8)
            if nx.is_connected(G):
                self.G = G
    
    def get_graph_collection(self):
        return self.graph_collection
    
    def get_done(self):
        return self.done
    
    def get_outputs(self):
        outputs = {}
        for i, agent in enumerate(self.agents):
            outputs[i] = agent.get_output()
        return outputs

    def run_next_round(self):
        if all(agent.done for agent in self.agents): # or self.current_round == 2*n -2:
            self.done = True
            print(">>> Mindenki kész, algoritmus LEÁLL <<<")
            return
        
        print(f"Round {self.current_round + 1}")
        self.generate_dynamic_graph()

        for i, agent in enumerate(self.agents):
            print(f"Agent {i}")
            neighbors = [self.agents[n] for n in self.G.neighbors(i)]
            agent.main(neighbors)
            print('-----------------------------------------------------------')

        ht_graphs = []
        # tree_fig = Figure(figsize=(len(self.agents)*3, len(self.agents)*1.5))  
        for i, agent in enumerate(self.agents):
            '''if i == 0:
                agent.myHT.draw_tree(i)
                agent.myHT_new.draw_tree(1)'''
            print(f"Agent {i}")
            agent.update_ht()
            # tree_ax = tree_fig.add_subplot(1, len(self.agents), i+1)
            # agent.myHT.draw_tree(i, tree_ax)
            ht_graphs.append(deepcopy(agent.myHT.get_tree()))

        self.graph_collection.add_next_round(deepcopy(self.G), ht_graphs)
        self.current_round += 1
