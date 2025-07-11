from collections import defaultdict
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import random
from algo import SimulationApp


class GraphViewer:
    def __init__(self):
        self.start()
        
    def start(self):
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
            self.which = num_zeros >= (n - num_zeros)
            random.shuffle(agents_inputs)

            root.deiconify()
            root.state("zoomed") 
            simulationApp = SimulationApp(n, agents_inputs)
            graphViewer = self.init_graphs(root, simulationApp, title)
            root.mainloop()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
            
    def init_graphs(self, root, simulation_app, title):
        self.top = None
        self.simulation_app = simulation_app
        self.root = root
        root.configure(bg="white")
        self.root.title(title)
        self.graph_frames = []
        self.graph_canvases = [None] * 4  # 4 frame-hez
        self.button_config()

        self.grid_frame = tk.Frame(root, bg="white")
        self.grid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        titles = [
            "Previous Agent Connections",
            "Previous History Trees",
            "Current Agent Connections",
            "Current History Trees"
        ]

        for i in range(2):
            for j in range(2):
                index = i * 2 + j
                title = titles[index]
                if j == 1:
                    frame = self.create_scrollable_frame(self.grid_frame, title, i, j)
                else:
                    frame = tk.LabelFrame(self.grid_frame, text=title, bd=1, relief=tk.SOLID, bg="white")
                    frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                    self.grid_frame.grid_rowconfigure(i, weight=1)
                    self.grid_frame.grid_columnconfigure(j, weight=1)
                self.graph_frames.append(frame)

    def create_scrollable_frame(self, parent, title, row, column):
        labelframe = tk.LabelFrame(parent, text=title, bd=1, relief=tk.SOLID, bg="white")
        labelframe.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")

        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(column, weight=1)
        canvas = tk.Canvas(labelframe, bg="white", bd=0,  highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar_x = tk.Scrollbar(
            labelframe,
            orient="horizontal",
            command=canvas.xview,
        )
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        canvas.configure(xscrollcommand=scrollbar_x.set)

        inner_frame = tk.Frame(canvas, bg="white")
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner_frame.bind("<Configure>", on_frame_configure)

        labelframe.grid_rowconfigure(0, weight=1)
        labelframe.grid_columnconfigure(0, weight=1)

        return inner_frame


    def draw_graphs_in_frame(self, frame_index, graphs, is_ht=False):
        frame = self.graph_frames[frame_index]
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        if self.graph_canvases[frame_index] is None:
            # Első megjelenítés
            if frame_index == 0 or frame_index == 2:
                fig, axs = plt.subplots(1, len(graphs), figsize=(len(graphs) * 1, 2))
            else:
                fig, axs = plt.subplots(1, len(graphs), figsize=(window_width / 185 * len(graphs) / 4, window_height / 250))
                fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0)

            if len(graphs) == 1:
                axs = [axs]

            for i, (g, ax) in enumerate(zip(graphs, axs)):
                ax.clear()
                if is_ht:
                    self.draw_tree(g, i, ax)
                else:
                    nx.draw(g, ax=ax, with_labels=True)

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.graph_canvases[frame_index] = (canvas, fig, axs)
        else:
            canvas, fig, _ = self.graph_canvases[frame_index]
            fig.clf()
            axs = fig.subplots(1, len(graphs))
            if len(graphs) == 1:
                axs = [axs]

            for i, (g, ax) in enumerate(zip(graphs, axs)):
                ax.clear()
                if is_ht:
                    self.draw_tree(g, i, ax)
                else:
                    nx.draw(g, ax=ax, with_labels=True)

            canvas.draw()
            self.graph_canvases[frame_index] = (canvas, fig, axs)


    def update_graphs(self):
        if self.simulation_app.get_done():
            return
        
        self.simulation_app.run_next_round()
        gc = self.simulation_app.get_graph_collection()

        if gc.G_prev:
            self.draw_graphs_in_frame(0, [gc.G_prev])
        if gc.HT_list_prev:
            self.draw_graphs_in_frame(1, gc.HT_list_prev, is_ht = True)
        if gc.G_curr:
            self.draw_graphs_in_frame(2, [gc.G_curr])
        if gc.HT_list_curr:
            self.draw_graphs_in_frame(3, gc.HT_list_curr, is_ht = True)

        if self.simulation_app.get_done():
            self.show_final_messages(self.simulation_app.get_outputs())

    def button_config(self):
        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.NONE)

        button_new = tk.Button(self.button_frame, text="New Simulation", command=self.restart_process)
        button_new.grid(row=0, column=0, padx=5, pady=5)

        button_step = tk.Button(self.button_frame, text="Step", command=self.update_graphs)
        button_step.grid(row=0, column=1, padx=5, pady=5)
        
        button_quit = tk.Button(self.button_frame, text="Exit", command=self.quit_process)
        button_quit.grid(row=0, column=2, padx=5, pady=5)

    def draw_tree(self, G, num, ax):
        try:
            # Create a consistent layout
            pos = {}
            levels = defaultdict(list)
            
            # Group nodes by level
            for node, data in G.nodes(data=True):
                levels[data['level']].append(node)
            
            # Assign positions level by level
            for level, nodes in sorted(levels.items()):
                y = -level  # Root at top (y=0), others below
                x_spacing = 1.0 / (len(nodes) + 1)
                for i, node in enumerate(sorted(nodes)):
                    x = (i + 1) * x_spacing
                    pos[node] = (x, y)
            
            # Ensure root is centered at top
            if 'Root' in G.nodes() and 'Root' not in pos:
                pos['Root'] = (0.5, 0)
            
            # Verify all nodes have positions
            missing_positions = [node for node in G.nodes() if node not in pos]
            if missing_positions:
                print(f"Warning: Missing positions for nodes: {missing_positions}")
                # Assign random positions to missing nodes
                for node in missing_positions:
                    pos[node] = (random.uniform(0,1), random.uniform(-len(levels),0))
            
            # Draw the graph
            ax.set_title(f'History Tree of {num}', fontsize=8, pad=0)
            ax.axis('off')
            
            # Separate edge types
            black_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('color') == 'black']
            red_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('color') == 'red']
            
            # Draw elements
            nx.draw_networkx_nodes(G, pos, node_size=75, node_color='lightblue', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color='black', width=1, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=1, style='dashed', ax=ax)
            
            # Draw labels
            node_labels = {
                n: ('R' if n == 'Root' else d.get('label', n)) 
                for n, d in G.nodes(data=True)
            }
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, ax=ax)
            
            # Add multiplicity labels for red edges
            red_edge_labels = {
                (u,v): str(d.get('multiplicity', 1)) 
                for u,v,d in G.edges(data=True) 
                if d.get('color') == 'red'
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=red_edge_labels, font_color='red', ax=ax, font_size=6)

            ax.margins(x=0.05, y=0)

        except Exception as e:
            print(f"Error drawing tree: {str(e)}")
            print("Current nodes:", list(G.nodes(data=True)))
            print("Current edges:", list(G.edges(data=True)))

    def show_final_messages(self, outputs):
        self.top = tk.Toplevel(self.root)
        self.top.title("Results")

        message_frame = tk.Frame(self.top, bg="white", bd=1)
        message_frame.pack(fill=tk.BOTH, expand=True)

        header_style = ttk.Style()
        header_style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))

        ttk.Label(message_frame, text="Agent", style="Header.TLabel").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(message_frame, text="Freq input", style="Header.TLabel").grid(row=0, column=1, padx=5, pady=5)

        for col in range(2):
            message_frame.grid_columnconfigure(col, weight=1)

        for i, (agent_id, freqs) in enumerate(outputs.items(), start=1):
            print(freqs)
            ttk.Label(message_frame, text=str(agent_id), font=("Helvetica", 14)).grid(row=i, column=0, padx=5, pady=2)
            freq = 0 if freqs.get(0, 0.0) > freqs.get(1, 0.0) else 1
            ttk.Label(message_frame, text=f"{0 if self.which else 1}", font=("Helvetica", 14)).grid(row=i, column=1, padx=5, pady=2)

        restart_button = ttk.Button(self.top, text="New Simulation", command=self.restart_process, style="TButton")
        restart_button.pack(pady=20)

    def restart_process(self):
        if self.root.winfo_exists():
            self.root.quit()
            self.root.destroy()
        self.start()

    def quit_process(self):
        if self.root.winfo_exists():
            self.root.quit()
            self.root.destroy()