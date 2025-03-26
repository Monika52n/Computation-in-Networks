import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk

# HistoryTree class
class HistoryTree:
    def __init__(self, input):
        self.G = nx.DiGraph()  # Directed Graph for History Tree
        self.create_initial_tree(input)

    def create_initial_tree(self, input):
        # Létrehozza az alap history tree-t, két csúccsal: root és Input
        self.G.add_node('root', label='Root')
        self.G.add_node('Input', label=input)
        self.G.add_edge('root', 'Input')
    
    def add_red_edge(self, node_from, node_to):
        self.G.add_edge(node_from, node_to)

    def add_bottom(self, input):
        pass

    def get_bottom():
        pass

    def chop(self):
        # A fában lévő szint levágása (szimuláció)
        #nodes_to_remove = [node for node, attr in self.G.nodes(data=True) if attr['label'] == level]
        #self.G.remove_nodes_from(nodes_to_remove)
        pass

    def compute_frequencies(self):
        # Itt pl. egy egyszerű számítás lehetne, hogy hány "Input" van a fában
        input_count = sum(1 for node, attr in self.G.nodes(data=True) if attr['label'] == 'Input')
        return input_count
    
    def visualize(self):
        # Gráf vizualizálása matplotlib segítségével
        pos = nx.spring_layout(self.G)  # Elhelyezési algoritmus
        labels = nx.get_node_attributes(self.G, 'label')
        nx.draw(self.G, pos, with_labels=True, node_size=2000, node_color='skyblue')
        nx.draw_networkx_labels(self.G, pos, labels=labels)
        plt.show()

    def get_max_height(self):
        return nx.dag_longest_path_length(self.G, 'root') if self.G.nodes else 0
