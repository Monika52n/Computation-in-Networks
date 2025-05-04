from history_tree import HistoryTree
from collections import defaultdict
import numpy as np
from copy import deepcopy
from scipy.optimize import nnls

class Agent:
    def __init__(self, n, input_value):
        self.n = n  # Number of agents in the network
        self.input_value = input_value  # Agent's input
        self.receivedMessages = []
        self.myHT = HistoryTree('Root', self.input_value)
        self.myHT_new = deepcopy(self.myHT)
        self.done = False
        self.output_m = None

    def input(self):
        return self.input_value
    
    def get_output(self):
        return self.output_m

    def output(self, message):
        self.output_m = message
        print(f"Agent output: {message}")

    def send_to_neighbor(self):
        return self.myHT

    def receive_from_neighbor(self, receivedMessage):
        self.receivedMessages.append(receivedMessage)

    def chop(self, history_tree):
        history_tree.chop()

    def compute_frequencies(self, history_tree):
        # 1. Szintek összegyűjtése
        level_nodes = defaultdict(list)
        for node, data in history_tree.G.nodes(data=True):
            level = data.get('level')
            if level is not None:
                level_nodes[level].append(node)

        counting_level = None
        candidate_levels = sorted(k for k in level_nodes if k >= 0)

        for level in candidate_levels:
            current = level_nodes[level]
            next_nodes = level_nodes.get(level + 1, [])

            # Feltétel: minden csúcsnak pontosan 1 black gyereke legyen
            if all(
                sum(1 for succ in history_tree.G.successors(node)
                    for k in history_tree.G[node][succ]
                    if history_tree.G[node][succ][k].get('color') == 'black') == 1
                for node in current
            ):
                # További feltétel: legalább egy érvényes piros él kell legyen
                found_valid_red = False
                for u in current:
                    for v in next_nodes:
                        if history_tree.G.has_edge(u, v):
                            for k in history_tree.G[u][v]:
                                if history_tree.G[u][v][k].get('color') == 'red':
                                    found_valid_red = True
                                    break
                        if found_valid_red:
                            break
                    if found_valid_red:
                        break
                if found_valid_red:
                    counting_level = level
                    break

        if counting_level is None:
            print("No counting level found with valid red edges")
            return {}

        print(f"Counting level found: {counting_level}")

        # 2. Egyenletek generálása
        nodes = level_nodes[counting_level]
        node_index = {node: i for i, node in enumerate(nodes)}
        equations = []
        processed_pairs = set()

        for u in nodes:
            for v in history_tree.G.successors(u):
                for k in history_tree.G[u][v]:
                    if history_tree.G[u][v][k].get('color') != 'red':
                        continue
                    v_level = history_tree.G.nodes[v].get('level')
                    if v_level != counting_level + 1:
                        continue

                    # v szülője: legyen p, akinek black éle van v-re
                    black_parents = [
                        pred for pred in history_tree.G.predecessors(v)
                        if pred in node_index
                        for k2 in history_tree.G[pred][v]
                        if history_tree.G[pred][v][k2].get('color') == 'black'
                    ]
                    if not black_parents:
                        continue
                    p = black_parents[0]

                    # u black gyereke
                    u_black_child = next(
                        (succ for succ in history_tree.G.successors(u)
                        for k3 in history_tree.G[u][succ]
                        if history_tree.G[u][succ][k3].get('color') == 'black'),
                        None
                    )
                    if not u_black_child:
                        continue

                    # Keressünk piros élt p → u_black_child
                    m2 = None
                    if history_tree.G.has_edge(p, u_black_child):
                        for k4 in history_tree.G[p][u_black_child]:
                            edge = history_tree.G[p][u_black_child][k4]
                            if edge.get('color') == 'red':
                                m2 = edge.get('multiplicity', 1)
                                break

                    if m2 is None:
                        continue

                    m1 = history_tree.G[u][v][k].get('multiplicity', 1)
                    eq = np.zeros(len(nodes))
                    eq[node_index[u]] = m1
                    eq[node_index[p]] = -m2
                    if (u, p) not in processed_pairs:
                        equations.append(eq)
                        processed_pairs.add((u, p))
                        print(f"Equation: {m1}·a({u}) = {m2}·a({p})")

        if not equations:
            print("No valid red edge equations found")
            return {}

        # 3. Megoldjuk az egyenletrendszert
        A = np.vstack(equations)
        A = np.vstack([A, np.ones(len(nodes))])
        b = np.zeros(len(equations) + 1)
        b[-1] = 1.0  # Normalizálási feltétel

        try:
            x, _ = nnls(A, b)
            x = x / np.sum(x)
        except Exception as e:
            print("Failed to solve system:", e)
            return {}

        # 4. Címke szerinti aggregálás
        label_counts = defaultdict(float)
        for i, node in enumerate(nodes):
            label = history_tree.G.nodes[node].get('label')
            if label is not None:
                label_counts[label] += x[i]

        frequencies = {label: freq for label, freq in label_counts.items()}
        print("Frequencies:", frequencies)
        return frequencies


    def update_ht(self):
        self.myHT = deepcopy(self.myHT_new)

    def main(self, neighbors):
        self.other_readies = []

        print('MAX height: ', self.myHT.get_max_height())

        if self.myHT.get_max_height() > 2 * self.n - 2:
            self.myHT_new = deepcopy(HistoryTree('Root', self.input_value))

        for neighbor in neighbors:
            #neighbor.receive_from_neighbor(self.send_to_neighbor()) #sending current ht to all neighbors
            self.receive_from_neighbor(neighbor.send_to_neighbor()) #receiving ht from all neighbors

        print(f"Length of received messages {len(self.receivedMessages)}")
        print('Input value: ', self.input_value)

        allMessages = self.receivedMessages + [self.myHT]
        
        minHT = min(allMessages, key=lambda ht: ht.get_max_height())
        #print("min height: ", minHT.get_max_height())
        #print("height myHT_new before chop: ", self.myHT_new.get_max_height())
        while self.myHT_new.get_max_height() > 1 and self.myHT_new.get_max_height() > minHT.get_max_height():
            self.chop(self.myHT_new)
        #print("height myHT_new before add bottom: ", self.myHT_new.get_max_height())

        self.myHT_new.add_bottom(self.input_value)

        for HT in self.receivedMessages:
            while HT.get_max_height() > 1 and HT.get_max_height() > minHT.get_max_height():
                self.chop(HT)
                #print("height ht chop: ", HT.get_max_height())
            #print("height ht: ", HT.get_max_height())
            #print("height myHT_new: ", self.myHT_new.get_max_height())
            self.myHT_new.merge_trees(HT)

        if self.myHT_new.get_max_height() >= 2 * self.n - 1:
            self.chop(self.myHT_new)

        self.receivedMessages = []

        #self.myHT_new.draw_tree(self.input_value)

        frequencies = None
        try: 
            frequencies = self.compute_frequencies(self.myHT_new)
        except Exception as e:
            print("Couldn't compute frequencies") # type: ignore
        if frequencies:
            self.output(frequencies)
            self.ready = True
        else:
            self.ready = False

        # 9. Ha mindenki készen van, leállhatunk
        if self.ready: # and all(self.other_readies):
            print(">>> Agent kész")
            self.done = True

def test_compute_frequencies():
    ht1 = HistoryTree("Root")

    # Szintek létrehozása
    ht1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('t_0', {'label': 'A', 'level': 0}),
        ('u_0', {'label': 'B', 'level': 0}),
        ('p_0', {'label': 'D', 'level': 0}),
        ('h_0', {'label': 'C', 'level': 0}),

        ('b_1', {'label': 'A', 'level': 1}),
        ('e_1', {'label': 'B', 'level': 1}),
        ('c_1', {'label': 'A', 'level': 1}),
        ('g_1', {'label': 'C', 'level': 1}),

        ('s_2', {'label': 'A', 'level': 2}),
        ('s2_2', {'label': 'B', 'level': 2}),
        ('f_2', {'label': 'A', 'level': 2}),
        ('f2_2', {'label': 'C', 'level': 2}),

        ('x_3', {'label': 'A', 'level': 3}),
    ])

    ht1.G.add_edges_from([
        ('Root', 't_0', {'color': 'black'}),
        ('Root', 'u_0', {'color': 'black'}),
        ('Root', 'p_0', {'color': 'black'}),
        ('Root', 'h_0', {'color': 'black'}),

        ('t_0', 'b_1', {'color': 'black'}),
        ('u_0', 'e_1', {'color': 'black'}),
        ('p_0', 'c_1', {'color': 'black'}),
        ('h_0', 'g_1', {'color': 'black'}),

        ('b_1', 's_2', {'color': 'black'}),
        ('e_1', 's2_2', {'color': 'black'}),
        ('c_1', 'f_2', {'color': 'black'}),
        ('g_1', 'f2_2', {'color': 'black'}),

        ('s_2', 'x_3', {'color': 'black'}),
    ])

    # Piros élek
    ht1.G.add_edges_from([
    ("t_0", "e_1", {'color': 'red', 'multiplicity': 2}),
    ("u_0", "b_1", {'color': 'red', 'multiplicity': 1}),

    ("b_1", "s2_2", {'color': 'red', 'multiplicity': 1}),
    ("e_1", "s_2", {'color': 'red', 'multiplicity': 1})
    ])

    ht1.red_edges = {
        # Level 0 -> Level 1
        ("t_0", "e_1"): 2,  # A típusú küld 2 üzenetet B-nek
        ("u_0", "b_1"): 1,  # B típusú küld 1 üzenetet A-nak

        # Level 1 -> Level 2
        ("b_1", "s2_2"): 1,  # A küld B-nek
        ("e_1", "s_2"): 1,   # B küld A-nak

    }
    
    ht1.draw_tree(0)

    # Hívjuk meg a frequency számítót
    agent = Agent(n=5, input_value="Root")
    frequencies = agent.compute_frequencies(ht1)

    # Ellenőrizzük, hogy a visszatérési érték szótár-e
    if isinstance(frequencies, dict):
        print("\n--- Teszt: compute_frequencies ---")
        for label, freq in frequencies.items():
            print(f"Label: {label}, Frequency: {freq:.4f}")
    else:
        print("Hiba történt a frekvenciák számítása közben. A visszatérési érték nem szótár.")

#test_compute_frequencies()

def test_compute_frequencies2():
    ht1 = HistoryTree("Root")

    # Szintek létrehozása
    ht1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('t_0', {'label': 'A', 'level': 0}),
        ('u_0', {'label': 'B', 'level': 0}),

        ('b_1', {'label': 'A', 'level': 1}),
        ('e_1', {'label': 'B', 'level': 1}),

        ('s_2', {'label': 'A', 'level': 2}),
        ('s2_2', {'label': 'B', 'level': 2}),

        ('x_3', {'label': 'A', 'level': 3}),
    ])

    ht1.G.add_edges_from([
        ('Root', 't_0', {'color': 'black'}),
        ('Root', 'u_0', {'color': 'black'}),

        ('t_0', 'b_1', {'color': 'black'}),
        ('u_0', 'e_1', {'color': 'black'}),

        ('b_1', 's_2', {'color': 'black'}),
        ('e_1', 's2_2', {'color': 'black'}),

        ('s_2', 'x_3', {'color': 'black'}),
    ])

    # Piros élek
    ht1.G.add_edges_from([
    ("t_0", "e_1", {'color': 'red', 'multiplicity': 2}),
    ("u_0", "b_1", {'color': 'red', 'multiplicity': 1}),

    ("b_1", "s2_2", {'color': 'red', 'multiplicity': 1}),
    ("e_1", "s_2", {'color': 'red', 'multiplicity': 1})
    ])

    ht1.red_edges = {
        # Level 0 -> Level 1
        ("t_0", "e_1"): 2,  # A típusú küld 2 üzenetet B-nek
        ("u_0", "b_1"): 1,  # B típusú küld 1 üzenetet A-nak

        # Level 1 -> Level 2
        ("b_1", "s2_2"): 1,  # A küld B-nek
        ("e_1", "s_2"): 1,   # B küld A-nak

    }
    
    ht1.draw_tree(0)
    ht1.is_black_tree_connected(None)

    # Hívjuk meg a frequency számítót
    agent = Agent(n=5, input_value="Root")
    frequencies = agent.compute_frequencies(ht1)

    # Ellenőrizzük, hogy a visszatérési érték szótár-e
    if isinstance(frequencies, dict):
        print("\n--- Teszt: compute_frequencies ---")
        for label, freq in frequencies.items():
            print(f"Label: {label}, Frequency: {freq:.4f}")
    else:
        print("Hiba történt a frekvenciák számítása közben. A visszatérési érték nem szótár.")

#test_compute_frequencies2()
