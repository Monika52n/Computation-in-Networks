from history_tree import HistoryTree
from collections import defaultdict
import numpy as np
from copy import deepcopy

class Agent:

    def __init__(self, n, input_value):
        self.n = n  # Number of agents in the network
        self.input_value = input_value  # Agent's input
        self.receivedMessages = []
        self.myHT = HistoryTree('Root', self.input_value)
        self.myHT_new = deepcopy(self.myHT)
        self.done = False

    def input(self):
        return self.input_value

    def output(self, message):
        print(f"Agent output: {message}")

    def send_to_neighbor(self):
        return self.myHT

    def receive_from_neighbor(self, receivedMessage):
        self.receivedMessages.append(receivedMessage)

    def chop(self, history_tree):
        history_tree.chop()

    def compute_frequencies2(self, history_tree):
        """Compute input frequencies based on the first counting level found in the history tree"""
        # Step 1: Find all levels and group nodes by level
        level_nodes = defaultdict(list)
        for node, data in history_tree.G.nodes(data=True):
            level = data.get('level')
            if level is not None:
                level_nodes[level].append(node)

        # Step 2: Find the first counting level (where each node has exactly one BLACK child)
        counting_level = None
        for level in sorted(level_nodes.keys()):
            is_counting_level = True
            for node in level_nodes[level]:
                # Get all BLACK children in next level
                black_children = [c for c in history_tree.G.successors(node)
                            if history_tree.G[node][c].get('color') == 'black']
                if len(black_children) != 1:
                    is_counting_level = False
                    break

            if is_counting_level and level + 1 in level_nodes:
                counting_level = level
                break

        if counting_level is None:
            print("No counting level found in the history tree")
            return {}

        print(f"Counting level found: {counting_level}")

        # Step 3: Prepare data structures
        nodes = level_nodes[counting_level]
        node_index = {node: i for i, node in enumerate(nodes)}
        equations = []
        processed_pairs = set()

        # Step 4: Find all RED edges between counting level nodes
        for u in nodes:
            # Find all RED edges from u to next level
            for v in history_tree.G.successors(u):
                if history_tree.G[u][v].get('color') == 'red':
                    # Find the BLACK parent of v in counting level
                    black_parents = [p for p in history_tree.G.predecessors(v)
                                if history_tree.G[p][v].get('color') == 'black'
                                and p in nodes]

                    if not black_parents:
                        continue

                    p = black_parents[0]  # Should be exactly one black parent

                    # Skip if we've already processed this pair
                    if (u, p) in processed_pairs or (p, u) in processed_pairs:
                        continue

                    # Now find if there's a RED edge from p back to u's child
                    u_black_child = [c for c in history_tree.G.successors(u)
                                if history_tree.G[u][c].get('color') == 'black'][0]

                    red_edges_to_u_child = [e for e in history_tree.G.in_edges(u_black_child)
                                        if history_tree.G[e[0]][e[1]].get('color') == 'red'
                                        and e[0] in nodes]

                    if not red_edges_to_u_child:
                        continue

                    # Get multiplicities
                    m1 = history_tree.G[u][v].get('multiplicity', 1)
                    m2 = history_tree.G[red_edges_to_u_child[0][0]][red_edges_to_u_child[0][1]].get('multiplicity', 1)

                    # Create equation: m1*a(u) = m2*a(p)
                    equation = np.zeros(len(nodes))
                    equation[node_index[u]] = m1
                    equation[node_index[p]] = -m2
                    equations.append(equation)
                    processed_pairs.add((u, p))
                    print(f"Equation added: {m1}*a({u}) = {m2}*a({p})")

        if not equations:
            print("No valid equations could be formed from red edges")
            return {}

        # Step 5: Solve the equation system
        A = np.vstack(equations)
        print(f"Equation matrix:\n{A}")

        # Add sum constraint that all frequencies sum to 1
        sum_constraint = np.ones(len(nodes))
        A = np.vstack([A, sum_constraint])
        b = np.zeros(A.shape[0])
        b[-1] = 1  # The sum should equal 1

        try:
            # First try regular least squares
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            # If underdetermined or solution has negative values, use NNLS
            if rank < len(nodes) or np.any(x < 0):
                print("Using non-negative least squares for better solution")
                from scipy.optimize import nnls
                x, _ = nnls(A, b)

                if np.allclose(x, 0):
                    print("Could not find valid non-zero solution")
                    return {}

            # Normalize the solution
            x = np.maximum(x, 0)  # Ensure no negative values
            x = x / np.sum(x)  # Normalize to sum to 1

        except Exception as e:
            print(f"Failed to solve equations: {e}")
            return {}

        # Step 6: Compute frequencies
        label_counts = defaultdict(float)
        for i, node in enumerate(nodes):
            label = history_tree.G.nodes[node].get('label')
            if label is not None:
                label_counts[label] += x[i]

        # Normalize again in case of rounding errors
        total = sum(label_counts.values())
        if total <= 0:
            print("Invalid frequency sum")
            return {}

        frequencies = {label: count/total for label, count in label_counts.items()}
        print("Computed frequencies:", frequencies)
        return frequencies

    def compute_frequencies(self, history_tree):
        """Compute input frequencies based on the first counting level found in the history tree"""
        # Step 1: Find all levels and group nodes by level
        level_nodes = defaultdict(list)
        for node, data in history_tree.G.nodes(data=True):
            level = data.get('level')
            if level is not None:
                level_nodes[level].append(node)

        # Step 2: Find the first counting level (where each node has exactly one BLACK child)
        counting_level = None
        for level in sorted(level_nodes.keys()):
            is_counting_level = True
            for node in level_nodes[level]:
                # Get all BLACK children in next level (handling multigraph)
                black_children = []
                for successor in history_tree.G.successors(node):
                    for edge_key in history_tree.G[node][successor]:
                        if history_tree.G[node][successor][edge_key].get('color') == 'black':
                            black_children.append(successor)
                            break  # only need one black edge per child
                
                if len(set(black_children)) != 1:  # using set to handle possible duplicates
                    is_counting_level = False
                    break

            if is_counting_level and level + 1 in level_nodes:
                counting_level = level
                break

        if counting_level is None:
            print("No counting level found in the history tree")
            return {}

        print(f"Counting level found: {counting_level}")

        # Step 3: Prepare data structures
        nodes = level_nodes[counting_level]
        node_index = {node: i for i, node in enumerate(nodes)}
        equations = []
        processed_pairs = set()

        # Step 4: Find all RED edges between counting level nodes
        for u in nodes:
            # Find all RED edges from u to next level (handling multigraph)
            for v in history_tree.G.successors(u):
                for edge_key in history_tree.G[u][v]:
                    if history_tree.G[u][v][edge_key].get('color') == 'red':
                        # Find the BLACK parent of v in counting level
                        black_parents = []
                        for predecessor in history_tree.G.predecessors(v):
                            for pred_edge_key in history_tree.G[predecessor][v]:
                                if (history_tree.G[predecessor][v][pred_edge_key].get('color') == 'black'
                                and predecessor in nodes):
                                    black_parents.append(predecessor)
                                    break  # only need one black edge per parent

                        if not black_parents:
                            continue

                        p = black_parents[0]  # Should be exactly one black parent

                        # Skip if we've already processed this pair
                        if (u, p) in processed_pairs or (p, u) in processed_pairs:
                            continue

                        # Now find if there's a RED edge from p back to u's child
                        # Find u's black child (handling multigraph)
                        u_black_child = None
                        for successor in history_tree.G.successors(u):
                            for edge_key in history_tree.G[u][successor]:
                                if history_tree.G[u][successor][edge_key].get('color') == 'black':
                                    u_black_child = successor
                                    break
                            if u_black_child is not None:
                                break

                        if u_black_child is None:
                            continue

                        # Find red edges to u_black_child (handling multigraph)
                        red_edges_to_u_child = []
                        for predecessor in history_tree.G.predecessors(u_black_child):
                            for edge_key in history_tree.G[predecessor][u_black_child]:
                                if (history_tree.G[predecessor][u_black_child][edge_key].get('color') == 'red'
                                and predecessor in nodes):
                                    red_edges_to_u_child.append((predecessor, u_black_child))
                                    break  # only need one red edge per predecessor

                        if not red_edges_to_u_child:
                            continue

                        # Get multiplicities
                        m1 = history_tree.G[u][v][edge_key].get('multiplicity', 1)
                        m2_edge = red_edges_to_u_child[0]
                        m2 = history_tree.G[m2_edge[0]][m2_edge[1]][next(iter(history_tree.G[m2_edge[0]][m2_edge[1]]))].get('multiplicity', 1)

                        # Create equation: m1*a(u) = m2*a(p)
                        equation = np.zeros(len(nodes))
                        equation[node_index[u]] = m1
                        equation[node_index[p]] = -m2
                        equations.append(equation)
                        processed_pairs.add((u, p))
                        print(f"Equation added: {m1}*a({u}) = {m2}*a({p})")

        if not equations:
            print("No valid equations could be formed from red edges")
            return {}

        # Step 5: Solve the equation system
        A = np.vstack(equations)
        print(f"Equation matrix:\n{A}")

        # Add sum constraint that all frequencies sum to 1
        sum_constraint = np.ones(len(nodes))
        A = np.vstack([A, sum_constraint])
        b = np.zeros(A.shape[0])
        b[-1] = 1  # The sum should equal 1

        try:
            # First try regular least squares
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            # If underdetermined or solution has negative values, use NNLS
            if rank < len(nodes) or np.any(x < 0):
                print("Using non-negative least squares for better solution")
                from scipy.optimize import nnls
                x, _ = nnls(A, b)

                if np.allclose(x, 0):
                    print("Could not find valid non-zero solution")
                    return {}

            # Normalize the solution
            x = np.maximum(x, 0)  # Ensure no negative values
            x = x / np.sum(x)  # Normalize to sum to 1

        except Exception as e:
            print(f"Failed to solve equations: {e}")
            return {}

        # Step 6: Compute frequencies
        label_counts = defaultdict(float)
        for i, node in enumerate(nodes):
            label = history_tree.G.nodes[node].get('label')
            if label is not None:
                label_counts[label] += x[i]

        # Normalize again in case of rounding errors
        total = sum(label_counts.values())
        if total <= 0:
            print("Invalid frequency sum")
            return {}

        frequencies = {label: count/total for label, count in label_counts.items()}
        print("Computed frequencies:", frequencies)
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

        while len(self.myHT_new.get_path_to_root(self.myHT_new.bottom_node)) > 2 and self.myHT_new.get_max_height() > minHT.get_max_height():
            self.chop(self.myHT_new)

        self.myHT_new.add_bottom(self.input_value)

        for HT in self.receivedMessages:
            while HT.get_max_height() > 1 and HT.get_max_height() > minHT.get_max_height():
                self.chop(HT)
            self.myHT_new.merge_trees(HT)

        if self.myHT_new.get_max_height() == 2 * self.n - 1:
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
            self.output([(self.input_value, 100)])
            self.ready = False

        # 9. Ha mindenki készen van, leállhatunk
        if self.ready and all(self.other_readies):
            print(">>> Mindenki kész, algoritmus LEÁLL <<<")
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

# Az algoritmus akkor áll le, amikor minden ügynök (agent) ki tudja számolni a bemenetek gyakoriságát (compute_frequencies).

