from tkinter import messagebox, simpledialog
from matplotlib.figure import Figure
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

        # Create the canvas for scrolling content
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollbar2 = tk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.configure(xscrollcommand=self.scrollbar2.set)

        # Pack the scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar2.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.graph_frame = tk.Frame(self.canvas)
        self.canvas.create_window((20, 20), window=self.graph_frame, anchor="nw")
        self.next_button = tk.Button(self.canvas, text="Next round", command=self.run_next_round)
        self.next_button.pack()

    def generate_dynamic_graph(self):
        return nx.erdos_renyi_graph(self.n, p=0.5)

    def draw_graph(self, G):
        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.clear()
        nx.draw(G, ax=ax, with_labels=True, node_color="lightblue", edge_color="gray", node_size=800)

        # Add the figure to the graph_frame
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.get_tk_widget().pack(pady=10)  # Adds spacing between graphs
        canvas.draw()

    def run_next_round(self):
        if all(agent.done for agent in self.agents): # or self.current_round == 2*n -2:
            print(">>> Mindenki kész, algoritmus LEÁLL <<<")
            self.next_button.config(state=tk.DISABLED)
            return
        
        print(f"Round {self.current_round + 1}")
        G = self.generate_dynamic_graph()
        self.draw_graph(G)

        for i, agent in enumerate(self.agents):
            print(f"Agent {i}")
            neighbors = [self.agents[n] for n in G.neighbors(i)]
            agent.main(neighbors)
            print('-----------------------------------------------------------')


        tree_fig = Figure(figsize=(len(self.agents)*3, len(self.agents)*1.5))  
        for i, agent in enumerate(self.agents):
            print(f"Agent {i}")
            agent.update_ht()
            tree_ax = tree_fig.add_subplot(1, len(self.agents), i+1)
            agent.myHT.draw_tree(i, tree_ax)

        self.current_round += 1

        # Add the tree figure to the graph_frame
        tree_canvas = FigureCanvasTkAgg(tree_fig, master=self.graph_frame)
        tree_canvas.get_tk_widget().pack(pady=10)  # Adds spacing between trees
        tree_canvas.draw()

        # Update the scrollable region
        self.graph_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    try:
        n = simpledialog.askinteger("Input", "Number of agents:")
        if n is None or n <= 0:
            raise ValueError("Operation cancelled.")

        num_zeros = simpledialog.askinteger("Input", f"How many of the {n} agents should have the input of 0?")
        if num_zeros is None:
            raise ValueError("Operation cancelled.")
        if num_zeros > n or num_zeros < 0:
            raise ValueError("The number of zeros cannot be greater than n or less than 0.")

        agents_inputs = [0] * num_zeros + [1] * (n - num_zeros)
        random.shuffle(agents_inputs)

        root.deiconify()
        root.title("Universal self-stabilizing finite-state algorithm")
        root.state("zoomed") 
        app = SimulationApp(root, n, agents_inputs)
        root.mainloop()

    except ValueError as e:
        messagebox.showerror("Error", str(e))
