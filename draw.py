from collections import defaultdict
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import random


class GraphViewer:
    def __init__(self, root, simulation_app):
        self.simulation_app = simulation_app
        self.root = root
        self.root.title("Gráf néző")
        self.graph_frames = []

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_refresh_button()

        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for i in range(2):
            for j in range(2):
                frame = tk.Frame(self.grid_frame, bd=1, relief=tk.SOLID)
                frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                self.grid_frame.grid_rowconfigure(i, weight=1)
                self.grid_frame.grid_columnconfigure(j, weight=1)
                self.graph_frames.append(frame)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def draw_graphs_in_frame(self, frame, graphs, horizontal=True):
        fig, axs = plt.subplots(1, len(graphs), figsize=(4 * len(graphs), 4)) if horizontal else \
                   plt.subplots(len(graphs), 1, figsize=(5, 3 * len(graphs)))

        if len(graphs) == 1:
            axs = [axs]

        for g, ax in zip(graphs, axs):
            ax.clear()
            nx.draw(g, ax=ax, with_labels=True)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_graphs(self):
        self.simulation_app.run_next_round()
        gc = self.simulation_app.get_graph_collection()

        frames = self.graph_frames
        self.clear_frame(frames[0])
        self.clear_frame(frames[1])
        self.clear_frame(frames[2])
        self.clear_frame(frames[3])

        if gc.G_prev:
            self.draw_graphs_in_frame(frames[0], [gc.G_prev])
        if gc.G_curr:
            self.draw_graphs_in_frame(frames[2], [gc.G_curr])

        if gc.HT_list_prev:
            self.draw_graphs_in_frame(frames[1], gc.HT_list_prev)
        if gc.HT_list_curr:
            self.draw_graphs_in_frame(frames[3], gc.HT_list_curr)

    def create_refresh_button(self):
        button = tk.Button(self.button_frame, text="Következő lépés", command=self.update_graphs)
        button.pack(pady=5)

