from agent import Agent
from history_tree import HistoryTree
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class SimulationApp:
    def __init__(self, root, n, agents_inputs):
        self.root = root
        self.n = n
        self.agents_inputs = agents_inputs
        self.current_round = 0
        self.agents = [Agent(n, input_value) for input_value in agents_inputs]

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.next_button = tk.Button(root, text="Next round", command=self.run_next_round)
        self.next_button.pack()

        self.run_next_round()

    def generate_dynamic_graph(self):
        return nx.erdos_renyi_graph(self.n, p=0.5)

    def draw_graph(self, G):
        self.ax.clear()
        nx.draw(G, ax=self.ax, with_labels=True, node_color="lightblue", edge_color="gray", node_size=800)
        self.canvas.draw()

    def run_next_round(self):
        print(f"Round {self.current_round + 1}")
        G = self.generate_dynamic_graph()
        self.draw_graph(G)

        for i, agent in enumerate(self.agents):
            neighbors = [self.agents[n] for n in G.neighbors(i)]
            agent.main(neighbors)

        self.current_round += 1


if __name__ == "__main__":
    agents_inputs = [1, 0, 0, 1, 1, 0, 0, 1]
    n = len(agents_inputs)

    root = tk.Tk()
    root.title("Universal self-stabilizing finite-state algorithm")
    app = SimulationApp(root, n, agents_inputs)
    root.mainloop()
