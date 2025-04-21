from copy import deepcopy
from tkinter import messagebox, simpledialog
from matplotlib.figure import Figure
from agent import Agent
from history_tree import HistoryTree
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from draw import GraphViewer 
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
        self.G = nx.erdos_renyi_graph(self.n, p=0.5)
    
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
            print(f"Agent {i}")
            agent.update_ht()
            # tree_ax = tree_fig.add_subplot(1, len(self.agents), i+1)
            # agent.myHT.draw_tree(i, tree_ax)
            ht_graphs.append(deepcopy(agent.myHT.get_tree()))

        self.graph_collection.add_next_round(deepcopy(self.G), ht_graphs)
        self.current_round += 1

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x700")
    root.withdraw()

    try:
        title = "Universal self-stabilizing finite-state algorithm"
        n = simpledialog.askinteger(title, "Number of agents:")
        if n is None or n <= 0:
            raise ValueError("Operation cancelled.")

        num_zeros = simpledialog.askinteger(title, f"How many of the {n} agents should have the input of 0?")
        if num_zeros is None:
            raise ValueError("Operation cancelled.")
        if num_zeros > n or num_zeros < 0:
            raise ValueError("The number of zeros cannot be greater than n or less than 0.")

        agents_inputs = [0] * num_zeros + [1] * (n - num_zeros)
        random.shuffle(agents_inputs)

        root.deiconify()
        root.state("zoomed") 
        simulationApp = SimulationApp(n, agents_inputs)
        graphViewer = GraphViewer(root, simulationApp, title)
        root.mainloop()

    except ValueError as e:
        messagebox.showerror("Error", str(e))
