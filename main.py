import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk

# Gráf inicializálás
G = nx.Graph()
G.add_nodes_from(range(5))  # 5 csúcs

# Algoritmus állapota
step = 0
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # Előre definiált lépések
G.add_edge(1, 3)

def update_graph():
    global step
    if step < len(edges):  
        G.add_edge(*edges[step])  # Új él hozzáadása
        step += 1
        draw_graph()

def draw_graph():
    plt.clf()
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=800)
    plt.draw()

# Tkinter UI létrehozása
root = tk.Tk()
root.title("Gráf animáció")

tk.Button(root, text="Következő lépés", command=update_graph).pack()

plt.ion()  # Interaktív mód
fig = plt.figure()
draw_graph()
plt.show()

root.mainloop()
