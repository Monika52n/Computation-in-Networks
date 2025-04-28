import random
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import copy

class HistoryTree:
    def __init__(self, root_label, input_value):
        self.G = nx.MultiDiGraph()
        self.root = root_label
        self.G.add_nodes_from([
            (root_label, {'label': root_label, 'level': -1})
            ,(f'N_{input_value}', {'label': input_value, 'level': 0})
        ])
        self.G.add_edges_from([
            ('Root', f'N_{input_value}', {'color': 'black'})
        ])
        self.G.graph['Root'] = root_label
        self.bottom_node = f'N_{input_value}'
        self.current_level = 1
        self.red_edges = defaultdict(int)
        self.id = input_value
        #self.counter = 0
       
    """ def __init__(self, root_label):
        self.G = nx.MultiDiGraph()
        self.root = root_label
        self.G.add_nodes_from([
            (root_label, {'label': root_label, 'level': -1})
        ])
        self.G.graph['Root'] = root_label
        self.bottom_node = root_label
        self.current_level = -1
        self.red_edges = defaultdict(int)  """
        #self.counter = 0

    def _get_edge_if_exists(self, from_node, to_node, color):
        edges_data = self.G.get_edge_data(from_node, to_node)
        if edges_data is not None:
            for edge, attr in edges_data.items():
                if attr['color'] == color:
                    return edges_data[edge]
        return None

    def _increase_red_edge_multiplicity(self, from_node, to_node, edge, multiplicity=1):
        self.red_edges[(from_node, to_node)] += multiplicity
        edge['multiplicity'] += multiplicity

    def _add_red_edge(self, from_node, to_node, multiplicity=1):
        existing_edge = self._get_edge_if_exists(from_node, to_node, color='red')
        if existing_edge is not None:
            self._increase_red_edge_multiplicity(from_node, to_node, existing_edge, multiplicity)
        else:
            self.red_edges[(from_node, to_node)] += multiplicity
            self.G.add_edge(from_node, to_node, color='red', multiplicity=multiplicity)

    def _check_if_nodes_match(self, this_node, other_node, other_tree):
        this_label = self.G.nodes[this_node]['label']
        other_label = other_tree.G.nodes[other_node]['label']
        this_path = self.get_path_to_root(this_node)
        other_path = other_tree.get_path_to_root(other_node)
        return this_path == other_path and this_label == other_label

    def _combine_inbound_red_edges(self, this_node, other_node, other_tree):
        for other_in_edge in other_tree.G.in_edges(other_node, data=True):
            bool_red_edge_in_main_tree_exists = False

            for this_in_edge in self.G.in_edges(this_node, data=True):
                if this_in_edge[2]['color'] == 'red' and other_in_edge[2]['color'] == 'red':
                    if self._check_if_nodes_match(this_in_edge[0], other_in_edge[0], other_tree): #check if source nodes match
                        self._add_red_edge(this_in_edge[0], this_node, other_in_edge[2]['multiplicity'])
                        bool_red_edge_in_main_tree_exists = True

            if not bool_red_edge_in_main_tree_exists and other_in_edge[2]['color'] == 'red':
                for source_node in self.G.nodes():
                    if self._check_if_nodes_match(source_node, other_in_edge[0], other_tree):
                        self._add_red_edge(source_node, this_node, other_in_edge[2]['multiplicity'])


    def match_node_into_level(self, this_level_nodes, other_node, other_tree, node_map):
        matched = False
        for this_node in this_level_nodes:
            this_path = self.get_path_to_root(this_node)
            other_path = other_tree.get_path_to_root(other_node)

            if this_path == other_path:
                matched = True
                node_map[other_node] = this_node
                self._combine_inbound_red_edges(this_node, other_node, other_tree)
                break

        return matched

    def _find_child_with_label(self, parent_node, label):
        child = None
        for c in self.G.successors(parent_node):
            if self.G.nodes[c]['label'] == label:
                child = c
        return child


    def _mark_interaction_with_red_edge(self, other_tree):
        old_bottom_node = self.G.nodes[self.bottom_node]
        self.add_bottom(old_bottom_node['label'])

        for other_bottom_node_pair in self.G.nodes():
            bool_match = self._check_if_nodes_match(other_bottom_node_pair, other_tree.bottom_node, other_tree)
            if bool_match:
                self.red_edges[(other_bottom_node_pair, self.bottom_node)] += 1
                self.G.add_edge(other_bottom_node_pair, self.bottom_node, color='red', multiplicity=1)

    def merge_trees(self, other_tree):
        node_map = {}
        node_map[other_tree.G.graph['Root']] = self.G.graph['Root']

        # Létrehozott másolatok
        self_before_chop = copy.deepcopy(self)
        other_before_chop = copy.deepcopy(other_tree)

        # Piros élek összegyűjtése
        red_edges_self = []
        for u, v, d in self.G.edges(data=True):
            if d.get('color') == 'red':
                if u in self.G.nodes:
                    red_edges_self.append((u, v, d))
        red_edges_other = []
        for u, v, d in other_tree.G.edges(data=True):
            if d.get('color') == 'red':
                if u in other_tree.G.nodes:
                    red_edges_other.append((u, v, d))

        red_edges_nr = max(len(red_edges_self), len(red_edges_other))

        # Szintenként történő összevonás
        for level in range(-1, max(self.current_level, other_tree.current_level) + 1):
            this_level_nodes = [n for n, attr in self.G.nodes(data=True) if attr['level'] == level]
            other_level_nodes = [n for n, attr in other_tree.G.nodes(data=True) if attr['level'] == level]

            for other_node in other_level_nodes:
                if other_node not in node_map:
                    matched = self.match_node_into_level(this_level_nodes, other_node, other_tree, node_map)

                    if not matched:
                        other_attrs = other_tree.G.nodes[other_node]
                        other_parent = next(other_tree.G.predecessors(other_node), None)

                        if other_parent and other_parent in node_map:
                            this_parent = node_map[other_parent]
                            matching_child = self._find_child_with_label(this_parent, other_attrs['label'])
                            node_to_check_for_red_edge = None

                            if matching_child is not None:
                                node_map[other_node] = matching_child
                                node_to_check_for_red_edge = matching_child
                            else:
                                # Új csúcs hozzáadása
                                new_node = self._generate_unique_node_name(other_node)
                                if this_parent != new_node:
                                    child_level = self.G.nodes[this_parent]['level'] + 1

                                    self.G.add_node(new_node, label=other_attrs['label'], level=child_level)
                                    self.G.add_edge(this_parent, new_node, color='black')

                                    node_map[other_node] = new_node
                                    node_to_check_for_red_edge = new_node
                                else:
                                    print(f"WARNING: Self-loop detected and avoided: {this_parent} -> {new_node}")

                            # Piros élek összekapcsolása
                            if node_to_check_for_red_edge is not None:
                                self._combine_inbound_red_edges(node_to_check_for_red_edge, other_node, other_tree)
        
        new_bottom_node = None
        bottom_level = max([self.G.nodes[n]['level'] for n in self.G.nodes()])  # Legmagasabb szint keresése
    
        # Megkeressük a legalsó szintű csúcsot
        new_bottom_nodes = [n for n, attr in self.G.nodes(data=True) if attr['level'] == bottom_level]
        
        if new_bottom_nodes:
            # Ha van legalább egy legalsó szintű csúcs, akkor az lesz az új bottom node
            self.bottom_node = new_bottom_nodes[0]  # Válasszunk egyet a legalsó szintű csúcsok közül
            print(f"New bottom node set: {self.G.nodes[self.bottom_node]}")
        else:
            print("WARNING: No bottom node found after merge!")

        # Piros élek kezelése
        self._mark_interaction_with_red_edge(other_tree)

        # Ellenőrizzük, hogy a piros élek nem tűntek el
        red_edges_self_after = []
        for u, v, d in self.G.edges(data=True):
            if d.get('color') == 'red':
                if u in self.G.nodes:
                    red_edges_self_after.append((u, v, d))

        print("min red edge BEFORE merge: ", red_edges_nr)
        print("red edge AFTER merge: ", len(red_edges_self_after))
        if len(red_edges_self_after) < red_edges_nr:
            print("ERROR in MERGE: RED edge disappeared!!!")

        # Szinkronizálás és ellenőrzés
        if self.is_black_tree_connected(None):
            print("error in merge")
        if self.check_backward_black_edges():
            print("error in merge")

        return node_map


    def get_path_to_root(self, node):
        path = []
        # print('get_path_to_root: node: ', node)
        while node is not None:
            path.append(self.G.nodes[node]['label'])  # Az útvonalban a címkét tároljuk
            predecessors = list(self.G.predecessors(node))
            for edge in self.G.edges(data=True):
                for p in predecessors:
                    if edge[0] == p and edge[1] == node and edge[2]['color'] == 'red':
                        predecessors.remove(p)
            node = predecessors[0] if predecessors else None
        return list(reversed(path))  # A gyökértől induló sorrendben

    def draw_tree(self, num):
        try:
            # Create a consistent layout
            pos = {}
            levels = defaultdict(list)
            
            # Group nodes by level
            for node, data in self.G.nodes(data=True):
                levels[data['level']].append(node)
            
            # Assign positions level by level
            for level, nodes in sorted(levels.items()):
                y = -level  # Root at top (y=0), others below
                x_spacing = 1.0 / (len(nodes) + 1)
                for i, node in enumerate(sorted(nodes)):
                    x = (i + 1) * x_spacing
                    pos[node] = (x, y)
            
            # Ensure root is centered at top
            if 'Root' in self.G.nodes() and 'Root' not in pos:
                pos['Root'] = (0.5, 0)
            
            # Verify all nodes have positions
            missing_positions = [node for node in self.G.nodes() if node not in pos]
            if missing_positions:
                print(f"Warning: Missing positions for nodes: {missing_positions}")
                # Assign random positions to missing nodes
                for node in missing_positions:
                    pos[node] = (random.uniform(0,1), random.uniform(-len(levels),0))
            
            # Draw the graph
            plt.figure(figsize=(12, 8))
            
            # Separate edge types
            black_edges = [(u,v) for u,v,d in self.G.edges(data=True) if d.get('color') == 'black']
            red_edges = [(u,v) for u,v,d in self.G.edges(data=True) if d.get('color') == 'red']
            
            # Draw elements
            nx.draw_networkx_nodes(self.G, pos, node_size=700, node_color='lightblue')
            nx.draw_networkx_edges(self.G, pos, edgelist=black_edges, edge_color='black', width=2)
            nx.draw_networkx_edges(self.G, pos, edgelist=red_edges, edge_color='red', width=2, style='dashed')
            
            # Draw labels
            node_labels = {n: d.get('label', n) for n,d in self.G.nodes(data=True)}
            nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=10)
            
            # Add multiplicity labels for red edges
            red_edge_labels = {
                (u,v): str(d.get('multiplicity', 1)) 
                for u,v,d in self.G.edges(data=True) 
                if d.get('color') == 'red'
            }
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=red_edge_labels, font_color='red')
            
            #plt.title('History Tree Visualization %i' % self.id)
            plt.title(f'History Tree Visualization {num}')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error drawing tree: {str(e)}")
            print("Current nodes:", list(self.G.nodes(data=True)))
            print("Current edges:", list(self.G.edges(data=True)))

    def _tree_layout(self, node, x, y, dx):
        """Recursive tree layout using only black edges for hierarchy"""
        if node not in self.G:
            return {}

        pos = {node: (x, -y)}
        children = [
            child for child in self.G.successors(node) 
            if self.G[node][child].get('color') == 'black'  # Only follow black edges
        ]
        
        if not children:
            return pos
        
        step = dx / max(len(children), 1)
        child_x = x - (dx/2) + (step/2)

        for child in children:
            child_pos = self._tree_layout(child, child_x, y+1, step)
            pos.update(child_pos)
            child_x += step

        return pos
    
    # chop 
    '''def chop2(self):
        if len(self.get_path_to_root(self.bottom_node)) > 2:
            print("\n--- CHOP START ---")
            print("Tree before chop:")
            print("Nodes:", list(self.G.nodes(data=True)))
            print("Edges:", list(self.G.edges(data=True)))

            if not self.G.nodes():
                return

            # Step 1: Identify L0 nodes (direct children of root)
            l0_nodes = list(self.G.successors("Root"))

            # Step 2: Collect edges to preserve
            edges_to_preserve = {
                'black': defaultdict(int),
                'red': defaultdict(int)
            }

            for l0_node in l0_nodes:
                # Collect all edges from L0 nodes
                for _, neighbor, data in self.G.out_edges(l0_node, data=True):
                    edge_type = data.get('color', 'black')
                    edges_to_preserve[edge_type][("Root", neighbor)] += data.get('multiplicity', 1)

            # Step 3: Collect all nodes that need level updates
            nodes_to_update = {}
            for l0_node in l0_nodes:
                for node in nx.dfs_preorder_nodes(self.G, source=l0_node):
                    current_level = self.G.nodes[node]['level']
                    nodes_to_update[node] = current_level - 1 if current_level > 0 else current_level

            # Step 4: Remove L0 nodes
            self.G.remove_nodes_from(l0_nodes)

            # Step 5: Rebuild connections and update levels
            for node, new_level in nodes_to_update.items():
                if node in self.G.nodes:
                    self.G.nodes[node]['level'] = new_level
                    if new_level == 0:  # These become new L0 nodes
                        # Add black edge from root if not already exists
                        if not self.G.has_edge("Root", node):
                            self.G.add_edge("Root", node, color='black')


            # Step 6: Restore all edges
            for edge_type in ['black', 'red']: #'red'
                for (u, v), m in edges_to_preserve[edge_type].items():
                    if v in self.G.nodes:
                        if not self.G.has_edge(u, v):  # Avoid duplicate edges
                            self.G.add_edge(u, v, color=edge_type, multiplicity=m)


            # Step 7: Merge isomorphic nodes
            while self._merge_all_levels():
                pass
                
            print("--- CHOP END ---")
            print("Tree after chop:")
            print("Nodes:", list(self.G.nodes(data=True)))
            print("Edges:", list(self.G.edges(data=True)))
            print("--- END ---\n")'''
    
    def get_nodes_at_level(self, level):
        res = []
        for node, attr in self.G.nodes(data=True):
            if attr['level'] == level:
                res.append(node)
        return res

    def _shift_nodes_by_level(self, level):
        for node, attr in self.G.nodes(data=True):
            if node != "Root":
                self.G.nodes[node]['level'] -= level

    def chop(self):
        if len(self.get_path_to_root(self.bottom_node)) > 2:
            graph_before_chop = copy.deepcopy(self)
            red_edges = []
            for u, v, d in self.G.edges(data=True):
                if d.get('color') == 'red':
                    if u in self.G.nodes and self.G.nodes[u].get('level', -1) >= 1:
                        red_edges.append((u, v, d))
            
            self.G.remove_nodes_from(self.get_nodes_at_level(0))

            self._shift_nodes_by_level(1)
            self.current_level -= 1
            
            # Elmentett csúcsok chop után
            """ nodes_after_chop = set(self.G.nodes)

            # Piros élek visszaállítása, ha a két végpont még él
            for u, v, edge_data in red_edges:
                if u in nodes_after_chop and v in nodes_after_chop:
                    # Megvizsgáljuk, van-e már ilyen piros él
                    found = False
                    for k in self.G[u].get(v, {}):
                        if self.G[u][v][k].get("color") == "red":
                            # Már van ilyen él, frissítjük a multiplicity-t
                            self.G[u][v][k]["multiplicity"] += edge_data.get("multiplicity", 1)
                            found = True
                            break

                    if not found:
                        # Ha még nincs ilyen él, hozzáadjuk
                        self.G.add_edge(u, v,
                                        color="red",
                                        multiplicity=edge_data.get("multiplicity", 1)) """

            red_edges_after1 = []
            for u, v, d in self.G.edges(data=True):
                if d.get('color') == 'red':
                    red_edges_after1.append((u, v, d))

            red_edges_after1.sort()
            red_edges.sort()
            if red_edges != red_edges_after1:
                print("ERROR after shift: RED edge disappeared!!!")

            for new_l0_node in self.get_nodes_at_level(0):
                self.G.add_edge('Root', new_l0_node, color='black')

            red_edges_after2 = []
            for u, v, d in self.G.edges(data=True):
                if d.get('color') == 'red':
                    red_edges_after2.append((u, v, d))

            red_edges_after2.sort()
            red_edges.sort()
            if red_edges != red_edges_after2:
                print("ERROR after add_edge: RED edge disappeared!!!")
            while self._merge_all_levels():
                pass

            self.current_level = self.G.nodes[self.bottom_node]['level']
            
            red_edges_after = []
            for u, v, d in self.G.edges(data=True):
                if d.get('color') == 'red':
                    red_edges_after.append((u, v, d))

            print("RED edge after chop: ", len(red_edges_after))
            red_edges_after.sort()
            red_edges.sort()
            if red_edges != red_edges_after:
                print("ERROR: RED edge disappeared!!!")
                print("Tree before chop:")
                print("Nodes:", list(graph_before_chop.G.nodes(data=True)))
                print("Edges:", list(graph_before_chop.G.edges(data=True)))
                print("Red edge before:", red_edges)
                print("Red edge after:", red_edges_after)
                
            if self.is_black_tree_connected(graph_before_chop):
                print("error in chop")
            if self.check_backward_black_edges():
                print("error in chop")


    def is_black_tree_connected(self, before_chop):
        # 1. Fekete élekből építjük a gráfot
        black_edges = [
            (u, v) for u, v, k, d in self.G.edges(keys=True, data=True)
            if d.get("color") == "black"
        ]
        black_graph = nx.DiGraph()
        black_graph.add_nodes_from(self.G.nodes(data=True))
        black_graph.add_edges_from(black_edges)

        # 2. Megnézzük, hogy minden nem-root csúcs elérhető-e a 'Root'-ból
        nodes_except_root = [n for n in self.G.nodes if n != "Root"]

        unreachable = [n for n in nodes_except_root if not nx.has_path(black_graph, 'Root', n)]

        if unreachable:
            print("HIBA: A következő L0 csúcsok nem elérhetők a Root-ból (fekete éleken):", unreachable)
            print("Root is not connected to:", unreachable)
            if before_chop is not None:
                print("Tree before chop:")
                print("Nodes:", list(before_chop.G.nodes(data=True)))
                print("Edges:", list(before_chop.G.edges(data=True)))
        return unreachable

    def check_backward_black_edges(self):
        """
        Megvizsgálja, hogy van-e visszafelé mutató fekete él,
        azaz olyan fekete él, ahol a forrás szintje nagyobb, mint a cél szintje.
        """
        invalid_edges = []

        for u, v, data in self.G.edges(data=True):
            if data.get('color') == 'black':
                level_u = self.G.nodes[u].get('level', -2)
                level_v = self.G.nodes[v].get('level', -2)
                if level_u > level_v:
                    invalid_edges.append((u, v, level_u, level_v))

        if invalid_edges:
            print("❗ Visszafelé mutató fekete élek találhatók:")
            for u, v, lu, lv in invalid_edges:
                print(f"  {u} (level {lu}) → {v} (level {lv})")

        return invalid_edges


    '''def _safe_update_multiplicity(self, u, v, m):
        """Safely update edge multiplicity for any graph type"""
        try:
            # For MultiDiGraph (supports multiple edges between nodes)
            if hasattr(self.G, 'get_edge_data'):
                edge_data = self.G.get_edge_data(u, v)
                if edge_data:  # If edge exists
                    if isinstance(edge_data, dict):  # Multi-edge case
                        for key in edge_data:
                            if 'multiplicity' in edge_data[key]:
                                edge_data[key]['multiplicity'] += m
                            else:
                                edge_data[key]['multiplicity'] = m + 1
                        return True
                    else:  # Single edge case (DiGraph)
                        if 'multiplicity' in edge_data:
                            self.G[u][v]['multiplicity'] += m
                        else:
                            self.G[u][v]['multiplicity'] = m + 1
                        return True
        except:
            pass
        return False'''


    def _add_black_edge(self, source_node, target_node, **edge_data):
        if not self.G.has_edge(source_node, target_node):
            self.G.add_edge(source_node, target_node, **edge_data)

    def _combine_outbound_red_edges(self, source_node, target_node, multiplicity):
        rep_edges_data = self.G.get_edge_data(source_node, target_node)

        if rep_edges_data is not None:
            bool_black_edges_only = True
            for re in rep_edges_data:
                rep_edge_data = rep_edges_data[re]
                if rep_edge_data.get('color') == 'red':
                    self._add_red_edge(source_node, target_node, multiplicity)
                    bool_black_edges_only = False

            if not rep_edges_data or (rep_edges_data and bool_black_edges_only):
                self._add_red_edge(source_node, target_node, multiplicity)

    def _switch_outbound_edges(self, from_node, to_node):
        for child in list(self.G.successors(from_node)):
            edges_data = self.G.get_edge_data(from_node, child)
            for e in edges_data:
                edge_data = edges_data[e]

                if edge_data.get('color') == 'black':
                    self._add_black_edge(to_node, child, **edge_data)

                if edge_data.get('color') == 'red':
                    self._combine_outbound_red_edges(to_node, child, edge_data.get('multiplicity', 1))

    def _merge_nodes(self, representative, node):
        self._switch_outbound_edges(node, representative)

        self.G.remove_node(node)

    '''def _merge_nodes2(self, representative, node):
        """Merge node into representative including red edge handling"""
        # Redirect black edges (children)
        for child in list(self.G.successors(node)):
            edge_data = self.G.edges[node, child, 0]
            if edge_data.get('color') == 'black':
                if not self.G.has_edge(representative, child):
                    self.G.add_edge(representative, child, **edge_data)
        
        # Process inbound red edges (add multiplicities to representative)
        """ for predecessor in list(self.G.predecessors(node)):
            edge_data = self.G.edges[predecessor, node]
            if edge_data.get('color') == 'red':
                multiplicity = edge_data.get('multiplicity', 1)
                if self.G.has_edge(predecessor, representative):
                    # Combine multiplicities
                    self.G.edges[predecessor, representative]['multiplicity'] += multiplicity
                else:
                    # Create new red edge
                    self.G.add_edge(predecessor, representative, 
                                color='red', 
                                multiplicity=multiplicity) """
        
        # Process outbound red edges (add multiplicities)
        for child in list(self.G.successors(node)):
            edge_data = self.G.edges[node, child, 0]
            if edge_data.get('color') == 'red':
                multiplicity = edge_data.get('multiplicity', 1)
                if self.G.has_edge(representative, child):
                    self.G.edges[representative, child, 0]['multiplicity'] += multiplicity
                else:
                    self.G.add_edge(representative, child, 
                                color='red', 
                                multiplicity=multiplicity)
        
        # Finally remove the merged node
        self.G.remove_node(node)'''

    def _merge_all_levels(self):
        """Merge nodes with identical sub-views at all levels in a single pass"""
        merged = False
        levels = sorted(set(data.get('level', -2) for _, data in self.G.nodes(data=True)))
        for level in levels:
            if level < 0:
                continue
            merged |= self._merge_isomorphic_nodes_at_level(level)
        return merged

    def _merge_isomorphic_nodes_at_level(self, level):
        """Merge nodes with identical sub-views at a specific level"""
        level_nodes = [node for node, data in self.G.nodes(data=True) if data.get('level', -2) == level]
        if not level_nodes:
            return False

        groups = defaultdict(list)

        for node in level_nodes:
            groups[self._hash_sub_view(node)].append(node)

        merged = False
        for group in groups.values():
            if len(group) > 1:
                representative = group[0]
                for node in group[1:]:
                    if node == self.bottom_node:
                        self.bottom_node = representative
                    self._merge_nodes(representative, node)
                    merged = True

        return merged

    def _hash_sub_view(self, node):
        """Create a hashable representation of the sub-view rooted at node"""
        sub_view = {
            'label': self.G.nodes[node].get('label', ''),
            'children': []
        }
        
        for child in sorted(self.G.successors(node)):
            edge_data = self.G.edges[node, child, 0]
            if edge_data.get('color') == 'black':
                sub_view['children'].append(('black', self.G.nodes[child].get('label', '')))

        sub_view['children'].sort()
        return str(sub_view)
    
    def _generate_unique_node_name(self, base_name):
        #Ad egy egyedi node nevet, ha már létezik a base_name
        suffix = "m"
        candidate = f"{base_name}{suffix}"
        i = 1
        while candidate in self.G.nodes:
            candidate = f"{base_name}{suffix * i}"
            i += 1
        return candidate


    def add_bottom(self, input_value):
        if len(self.G.nodes) == 1:
            self.__init__('Root', input_value)
        else:
            current_bottom_node = self.G.nodes[self.bottom_node]
            # new_bottom_node = f"{self.bottom_node}_m"
            new_bottom_node = self._generate_unique_node_name(self.bottom_node)
            """ new_bottom_node = f"N_{self.counter}"
            self.counter += 1 """

            self.G.add_nodes_from([
                (new_bottom_node, {'label': input_value, 'level': current_bottom_node['level'] + 1})
            ])
            self.G.add_edge(self.bottom_node, new_bottom_node, color='black')
            self.bottom_node = new_bottom_node

            self.current_level = self.G.nodes[self.bottom_node]['level']

    def compute_frequencies(self):
        # Itt pl. egy egyszerű számítás lehetne, hogy hány "Input" van a fában
        input_count = sum(1 for node, attr in self.G.nodes(data=True) if attr['label'] == 'Input')
        return input_count

    def get_max_height(self):
        #return nx.dag_longest_path_length(self.G, 'Root') if self.G.nodes else 0
        return len(self.get_path_to_root(self.bottom_node)) - 1 #-1 for Root


################### tests merge
def test_merge_trees():
    print("Testing merge with predefined trees")
    
    ht1 = HistoryTree("Root")
    ht1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('B_0', {'label': '1', 'level': 0}),
        ('A_0', {'label': '0', 'level': 0}),
        ('C_1', {'label': '1', 'level': 1}),
        ('D_1', {'label': '1', 'level': 1}),
        ('E_1', {'label': '0', 'level': 1}),
        ('F_1', {'label': '0', 'level': 2})
    ])
    ht1.G.add_edges_from([
        ("Root", "A_0", {'color': 'black'}),
        ("Root", "B_0", {'color': 'black'}),
        ("A_0", "C_1", {'color': 'black'}),
        ("B_0", "D_1", {'color': 'black'}),
        ("B_0", "E_1", {'color': 'black'}),
        ("E_1", "F_1", {'color': 'black'}),
        ("A_0", "D_1", {'color': 'red', 'multiplicity': 1})
    ])
    ht1.bottom_node = "F_1"
    ht1.current_level = 2
    ht1.draw_tree(1)
    
    ht2 = HistoryTree("Root")
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A_3', {'label': '0', 'level': 0}),
        ('AA_3', {'label': '1', 'level': 0}),
        ('AAA_3', {'label': '1', 'level': 1}),
        ('B_3', {'label': '1', 'level': 1}),
        ('C_3', {'label': '0', 'level': 2})
    ])
    ht2.G.add_edges_from([
        ("Root", "A_3", {'color': 'black'}),
        ("Root", "AA_3", {'color': 'black'}),
        ("A_3", "B_3", {'color': 'black'}),
        ("AA_3", "AAA_3", {'color': 'black'}),
        ("B_3", "C_3", {'color': 'black'}),
        ("A_3", "AAA_3", {'color': 'red', 'multiplicity': 1}),
    ])
    ht2.bottom_node = "B_3"
    ht2.current_level = 2
    ht2.draw_tree(2)
    
    print("Merging trees...")

    ht1.merge_trees(ht2)
    ht1.draw_tree(1)

#test_merge_trees()

def test_complex_merge():
    print("Testing complex merge with deeper trees")
    
    ht1 = HistoryTree("Root")
    ht1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A', {'label': '0', 'level': 0}),
        ('B', {'label': '1', 'level': 0}),
        ('C', {'label': '1', 'level': 1}),
        ('D', {'label': '0', 'level': 1}),
        ('E', {'label': '1', 'level': 2}),
        ('F', {'label': '0', 'level': 2}),
        ('G', {'label': '1', 'level': 3})
    ])
    ht1.G.add_edges_from([
        ("Root", "A", {'color': 'black'}),
        ("Root", "B", {'color': 'black'}),
        ("A", "C", {'color': 'black'}),
        ("A", "D", {'color': 'black'}),
        ("C", "E", {'color': 'black'}),
        ("D", "F", {'color': 'black'}),
        ("E", "G", {'color': 'black'}),
    ])
    ht1.bottom_node = "G"
    ht1.current_level = 3
    ht1.draw_tree(1)
    
    ht2 = HistoryTree("Root")
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('H', {'label': '1', 'level': 0}),
        ('I', {'label': '0', 'level': 0}),
        ('J', {'label': '1', 'level': 1}),
        ('K', {'label': '1', 'level': 2}),
        ('L', {'label': '0', 'level': 2}),
        ('M', {'label': '1', 'level': 3}),
        ('N', {'label': '0', 'level': 3})
    ])
    ht2.G.add_edges_from([
        ("Root", "H", {'color': 'black'}),
        ("Root", "I", {'color': 'black'}),
        ("H", "J", {'color': 'black'}),
        ("J", "K", {'color': 'black'}),
        ("I", "L", {'color': 'black'}),
        ("K", "M", {'color': 'black'}),
        ("L", "N", {'color': 'black'}),

    ])
    ht2.bottom_node = "M"
    ht2.current_level = 3
    ht2.draw_tree(2)
    
    print("Merging trees...")
    ht1.merge_trees(ht2)
    ht1.draw_tree(1)

#test_complex_merge()

################### tests chop
def test_ht2_chop():
    ht2 = HistoryTree("Root")  # This will create root node named 'root' with label 'Root'
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('H', {'label': '1', 'level': 0}),
        ('I', {'label': '0', 'level': 1}),
        ('J', {'label': '0', 'level': 1}),
        ('K', {'label': '0', 'level': 2}),
        ('L', {'label': '0', 'level': 3}),
        ('M', {'label': '0', 'level': 4}),
        ('N', {'label': '0', 'level': 5})
    ])
    ht2.G.add_edges_from([
        ("Root", "H", {'color': 'black'}),
        ("H", "I", {'color': 'black'}),
        ("H", "J", {'color': 'black'}),
        ("J", "K", {'color': 'black'}),
        ("K", "L", {'color': 'black'}),
        ("L", "M", {'color': 'black'}),
        ("M", "N", {'color': 'black'}),
        ("I", "K", {'color': 'red', 'multiplicity': 1}),
        ("I", "L", {'color': 'red', 'multiplicity': 1}),
        ("I", "M", {'color': 'red', 'multiplicity': 1}),
        ("M", "N", {'color': 'red', 'multiplicity': 2}),

    ])
    ht2.bottom_node = "N"
    ht2.current_level = 5

    ht2.draw_tree(1)
    
    ht2.chop()
    
    print("\nDetailed Node Information (After Chop):")
    for node, data in ht2.G.nodes(data=True):
        print(f"Node {node}: label={data['label']}, level={data['level']}")
    
    print("\nDetailed Edge Information (After Chop):")
    for u, v, data in ht2.G.edges(data=True):
        edge_type = "BLACK" if data.get('color') == 'black' else "red"
        mult = data.get('multiplicity', 1)
        print(f"{edge_type} edge {u} -> {v} (multiplicity={mult})")
    
    print("\nAfter chop:")
    ht2.draw_tree(1)

# Run the test
def _safe_draw_tree(G):
    """Safe visualization that handles missing nodes"""
    try:
        pos = {}
        for node, data in G.nodes(data=True):
            level = data.get('level', -2)
            nodes_in_level = [n for n, d in G.nodes(data=True) if d.get('level', -2) == level]
            x_pos = nodes_in_level.index(node) if node in nodes_in_level else 0
            pos[node] = (level, -x_pos)
        
        edge_colors = ['red' if d.get('color') == 'red' else 'black' 
                      for _, _, d in G.edges(data=True)]
        
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_size=800)
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        print("Current nodes:", list(G.nodes()))
        print("Current edges:", list(G.edges()))

# Run the test
#test_linear_tree_chop()


def test_red_edges():
    print("Testing merge with predefined trees")

    ht1 = HistoryTree("Root")
    ht1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A_0', {'label': '0', 'level': 0})
    ])
    ht1.G.add_edges_from([
        ("Root", "A_0", {'color': 'black'})
    ])
    ht1.bottom_node = "A_0"
    ht1.current_level = 1
    ht1.draw_tree(1)

    ht2 = HistoryTree("Root")
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('B_0', {'label': '1', 'level': 0})
    ])
    ht2.G.add_edges_from([
        ("Root", "B_0", {'color': 'black'})
    ])
    ht2.bottom_node = "B_0"
    ht2.current_level = 1
    ht2.draw_tree(2)

    print("Merging trees...")
    ht2.merge_trees(ht1)
    ht2.draw_tree(2)

#test_red_edges()


def test_chop():
    ht2 = HistoryTree("Root")  # This will create root node named 'root' with label 'Root'
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A', {'label': '0', 'level': 0}),
        ('B', {'label': '1', 'level': 0}),
        ('C', {'label': '2', 'level': 0}),

        ('D', {'label': '0', 'level': 1}),
        ('E', {'label': '1', 'level': 1}),
        ('F', {'label': '1', 'level': 1}),

        ('G', {'label': '0', 'level': 2}),
        ('H', {'label': '1', 'level': 2}),
        ('I', {'label': '1', 'level': 2}),

        ('J', {'label': '0', 'level': 3})

    ])
    ht2.G.add_edges_from([
        ("Root", "A", {'color': 'black'}),
        ("Root", "B", {'color': 'black'}),
        ("Root", "C", {'color': 'black'}),

        ("A", "D", {'color': 'black'}),
        ("B", "E", {'color': 'black'}),
        ("B", "F", {'color': 'black'}),

        ("D", "G", {'color': 'black'}),
        ("E", "H", {'color': 'black'}),
        ("F", "I", {'color': 'black'}),

        ("G", "J", {'color': 'black'}),

        ###
        ("J", "I", {'color': 'black'}),

        ("A", "E", {'color': 'red', 'multiplicity': 1}),
        ("B", "D", {'color': 'red', 'multiplicity': 1}),
        ("C", "E", {'color': 'red', 'multiplicity': 1}),
        ("C", "F", {'color': 'red', 'multiplicity': 1}),

        ("E", "G", {'color': 'red', 'multiplicity': 1}),
        ("E", "H", {'color': 'red', 'multiplicity': 2}),
        ("E", "I", {'color': 'red', 'multiplicity': 1}),
        ("F", "I", {'color': 'red', 'multiplicity': 1}),

        ("H", "J", {'color': 'red', 'multiplicity': 1}),
        ("I", "J", {'color': 'red', 'multiplicity': 2}),

    ])
    ht2.bottom_node = "J"
    ht2.current_level = 4


    print("Before chop:")
    ht2.draw_tree(2)

    #ht2._merge_nodes('E', 'F')

    ht2.chop()

    ht2.draw_tree(2)

# Run the test
# test_chop()


def test_merge_trees2():
    print("Testing merge with predefined trees")

    ht1 = HistoryTree("Root")
    ht1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A1', {'label': '1', 'level': 0}),
        ('B1', {'label': '0', 'level': 0}),
        ('C1', {'label': '1', 'level': 1}),
    ])
    ht1.G.add_edges_from([
        ("Root", "A1", {'color': 'black'}),
        ("Root", "B1", {'color': 'black'}),
        ("A1", "C1", {'color': 'black'}),
        ("B1", "C1", {'color': 'red', 'multiplicity': 2}),
    ])
    ht1.bottom_node = "C1"
    ht1.current_level = 1
    ht1.draw_tree(1)

    ht2 = HistoryTree("Root")
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A2', {'label': '1', 'level': 0}),
        ('B2', {'label': '0', 'level': 0}),
        ('C2', {'label': '0', 'level': 1})

    ])
    ht2.G.add_edges_from([
        ("Root", "A2", {'color': 'black'}),
        ("Root", "B2", {'color': 'black'}),
        ("B2", "C2", {'color': 'black'}),
    ])
    ht2.bottom_node = "C2"
    ht2.current_level = 1
    ht2.draw_tree(2)

    print("Merging trees...")
    ht1.merge_trees(ht2)
    ht1.draw_tree(2)

#test_merge_trees2()


def test_max_height():
    ht2 = HistoryTree("Root")  # This will create root node named 'root' with label 'Root'
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A', {'label': '0', 'level': 0}),
        ('B', {'label': '1', 'level': 0}),
        ('C', {'label': '2', 'level': 0}),

        ('D', {'label': '0', 'level': 1}),
        ('E', {'label': '1', 'level': 1}),
        ('F', {'label': '1', 'level': 1}),

        ('G', {'label': '0', 'level': 2}),
        ('H', {'label': '1', 'level': 2}),
        ('I', {'label': '1', 'level': 2}),

        ('J', {'label': '0', 'level': 3})

    ])
    ht2.G.add_edges_from([
        ("Root", "A", {'color': 'black'}),
        ("Root", "B", {'color': 'black'}),
        ("Root", "C", {'color': 'black'}),

        ("A", "D", {'color': 'black'}),
        ("B", "E", {'color': 'black'}),
        ("B", "F", {'color': 'black'}),

        ("D", "G", {'color': 'black'}),
        ("E", "H", {'color': 'black'}),
        ("F", "I", {'color': 'black'}),

        ("G", "J", {'color': 'black'}),

        ###
        ("J", "I", {'color': 'black'}),

        ("A", "E", {'color': 'red', 'multiplicity': 1}),
        ("B", "D", {'color': 'red', 'multiplicity': 1}),
        ("C", "E", {'color': 'red', 'multiplicity': 1}),
        ("C", "F", {'color': 'red', 'multiplicity': 1}),

        ("E", "G", {'color': 'red', 'multiplicity': 1}),
        ("E", "H", {'color': 'red', 'multiplicity': 2}),
        ("E", "I", {'color': 'red', 'multiplicity': 1}),
        ("F", "I", {'color': 'red', 'multiplicity': 1}),

        ("H", "J", {'color': 'red', 'multiplicity': 1}),
        ("I", "J", {'color': 'red', 'multiplicity': 2}),

    ])
    ht2.bottom_node = "J"
    ht2.current_level = 4

    ht2.draw_tree(2)

    ht2.chop()

    ht2.draw_tree(1)
    print('Max height: ', ht2.get_max_height())
    print('Current level: ', ht2.current_level)
    print('Bottom node level: ', ht2.G.nodes[ht2.bottom_node]['level'])

#test_max_height()


def test_path_to_root():
    ht2 = HistoryTree("Root")  # This will create root node named 'root' with label 'Root'
    ht2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('A', {'label': '0', 'level': 0}),
        ('B', {'label': '1', 'level': 1}),

        ('C', {'label': '1', 'level': 0}),
    ])
    ht2.G.add_edges_from([
        ("Root", "A", {'color': 'black'}),
        ("Root", "C", {'color': 'black'}),

        #("A", "D", {'color': 'black'}),
        ("A", "B", {'color': 'red', 'multiplicity': 1}),
        ("C", "B", {'color': 'black'}),

    ])
    ht2.bottom_node = "B"
    ht2.current_level = 1

    ht2.draw_tree(2)

    print('Path to root: ', ht2.get_path_to_root(ht2.bottom_node))

    print('Max height: ', ht2.get_max_height())
    print('Current level: ', ht2.current_level)
    print('Bottom node level: ', ht2.G.nodes[ht2.bottom_node]['level'])

#test_path_to_root()

# test chop
def test_chop_complex_graph():
    ht = HistoryTree("Root")

    # Csomópontok hozzáadása
    ht.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('N_0', {'label': 0, 'level': 0}),
        ('N_0_m', {'label': 0, 'level': 1}),
        ('N_0_m_m', {'label': 0, 'level': 2}),
        ('N_0_m_m_m', {'label': 0, 'level': 3}),
        ('N_0_m_m_m_m', {'label': 0, 'level': 4}),
        ('N_1_m_m_m_m_m_m', {'label': 1, 'level': 0}),
        ('N_1_m_m_m_m_m_m_m', {'label': 1, 'level': 3}),
        ('N_1_m_m_m_m_m_m_m_m_m', {'label': 1, 'level': 2}),
        ('N_1_m_m_m_m_m_m_m_m_m_m_m', {'label': 1, 'level': 1})
    ])

    # Élek (fekete + piros) hozzáadása
    ht.G.add_edges_from([
        ('Root', 'N_0', {'color': 'black'}),
        ('Root', 'N_1_m_m_m_m_m_m', {'color': 'black'}),
        ('N_0', 'N_0_m', {'color': 'black'}),
        ('N_0_m', 'N_0_m_m', {'color': 'black'}),
        ('N_0_m_m', 'N_0_m_m_m', {'color': 'black'}),
        ('N_0_m_m_m', 'N_0_m_m_m_m', {'color': 'black'}),

        ('N_0', 'N_1_m_m_m_m_m_m_m', {'color': 'red', 'multiplicity': 2}),
        ('N_0', 'N_1_m_m_m_m_m_m_m_m_m', {'color': 'red', 'multiplicity': 1}),
        ('N_0_m', 'N_1_m_m_m_m_m_m_m', {'color': 'red', 'multiplicity': 1}),
        ('N_1_m_m_m_m_m_m', 'N_0', {'color': 'red', 'multiplicity': 2}),
        ('N_1_m_m_m_m_m_m', 'N_0_m_m', {'color': 'red', 'multiplicity': 1}),
        ('N_1_m_m_m_m_m_m', 'N_1_m_m_m_m_m_m_m_m_m', {'color': 'red', 'multiplicity': 2}),
        ('N_1_m_m_m_m_m_m_m', 'N_0', {'color': 'red', 'multiplicity': 1}),
        ('N_1_m_m_m_m_m_m', 'N_1_m_m_m_m_m_m_m', {'color': 'black'}),
        ('N_1_m_m_m_m_m_m_m', 'N_1_m_m_m_m_m_m_m_m_m', {'color': 'black'}),
        ('N_1_m_m_m_m_m_m_m_m_m', 'N_1_m_m_m_m_m_m_m_m_m_m_m', {'color': 'black'}),
    ])

    # Bottom node és szint beállítása
    ht.bottom_node = 'N_0_m_m_m_m'
    ht.current_level = 4

    # Chop előtt
    print("Before chop:")
    #ht.draw_tree(1)

    # Chop meghívása
    ht.chop()

    # Chop után
    print("After chop:")
    ht.draw_tree(2)

#test_chop_complex_graph()

def test_merge_trees_on_complex_graphs():
    # --- Setup 1 ---
    tree1 = HistoryTree("Root")
    tree1.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('N_1', {'label': 1, 'level': 0}),
        ('N_1m', {'label': 1, 'level': 1}),
    ])
    tree1.G.add_edges_from([
        ('Root', 'N_1', {'color': 'black'}),
        ('N_1', 'N_1m', {'color': 'black'}),
    ])
    tree1.bottom_node = 'N_1m'
    tree1.current_level = 1

    tree2 = HistoryTree("Root")
    tree2.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('N_1mmmmmm', {'label': 1, 'level': 0}),
        ('N_1mmmmmmm', {'label': 1, 'level': 1}),
        ('N_1mmmmmmmm', {'label': 1, 'level': 2}),
        ('N_1mmmmmmmmm', {'label': 1, 'level': 3}),
        ('N_1mmmmmmmmmm', {'label': 1, 'level': 4}),
        ('N_1mmmmmmmmmmm', {'label': 1, 'level': 5}),
        ('N_1mmmmmmmmmmmm', {'label': 1, 'level': 6}),
        ('N_0mmmmmm', {'label': 0, 'level': 0}),
        ('N_0mmmmmmm', {'label': 0, 'level': 1}),
        ('N_0mmmmmmmm', {'label': 0, 'level': 2}),
    ])
    tree2.G.add_edges_from([
        ('Root', 'N_1mmmmmm', {'color': 'black'}),
        ('N_1mmmmmm', 'N_1mmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmm', 'N_1mmmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmmm', 'N_1mmmmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmmmm', 'N_1mmmmmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmmmmm', 'N_1mmmmmmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmmmmmm', 'N_1mmmmmmmmmmmm', {'color': 'black'}),

        ('Root', 'N_0mmmmmm', {'color': 'black'}),
        ('N_0mmmmmm', 'N_0mmmmmmm', {'color': 'black'}),
        ('N_0mmmmmmm', 'N_0mmmmmmmm', {'color': 'black'}),

        ('N_1mmmmmmmm', 'N_1mmmmmmmmmmmm', {'color': 'red', 'multiplicity': 1}),
        ('N_0mmmmmmmm', 'N_1mmmmmmmmmm', {'color': 'red', 'multiplicity': 1}),
        ('N_0mmmmmmmm', 'N_1mmmmmmmmmmm', {'color': 'red', 'multiplicity': 1}),
    ])
    tree2.bottom_node = 'N_1mmmmmmmmmmmm'
    tree2.current_level = 6

    print("\nBEFORE MERGE:")
   # tree1.draw_tree(1)
    #tree2.draw_tree(2)

    # --- Merge ---
    tree1.merge_trees(tree2)

    print("\nAFTER MERGE:")
    tree1.draw_tree(1)


#test_merge_trees_on_complex_graphs()

def test_merge_trees_second_case():
    # --- Self tree ---
    self_tree = HistoryTree("Root")
    self_tree.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('N_1', {'label': 1, 'level': 0}),
        ('N_1m', {'label': 1, 'level': 1}),
        ('N_1mm', {'label': 1, 'level': 2}),
        ('N_0mmmmmmm', {'label': 0, 'level': 0}),
        ('N_0mmmmmmmm', {'label': 0, 'level': 1}),
    ])
    self_tree.G.add_edges_from([
        ('Root', 'N_1', {'color': 'black'}),
        ('N_1', 'N_1m', {'color': 'black'}),
        ('N_1m', 'N_1mm', {'color': 'black'}),
        ('Root', 'N_0mmmmmmm', {'color': 'black'}),
        ('N_0mmmmmmm', 'N_0mmmmmmmm', {'color': 'black'}),
    ])
    self_tree.bottom_node = 'N_1mm'
    self_tree.current_level = 2

    # --- Other tree ---
    other_tree = HistoryTree("Root")
    other_tree.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('N_0mmmmm', {'label': 0, 'level': 0}),
        ('N_0mmmmmm', {'label': 0, 'level': 1}),
        ('N_0mmmmmmm', {'label': 0, 'level': 2}),
        ('N_0mmmmmmmm', {'label': 0, 'level': 3}),
        ('N_0mmmmmmmmm', {'label': 0, 'level': 4}),
        ('N_0mmmmmmmmmm', {'label': 0, 'level': 5}),
        ('N_0mmmmmmmmmmm', {'label': 0, 'level': 6}),
        ('N_1mmmmmmm', {'label': 1, 'level': 0}),
        ('N_1mmmmmmmm', {'label': 1, 'level': 1}),
        ('N_1mmmmmmmmm', {'label': 1, 'level': 2}),
    ])
    other_tree.G.add_edges_from([
        ('Root', 'N_0mmmmm', {'color': 'black'}),
        ('N_0mmmmm', 'N_0mmmmmm', {'color': 'black'}),
        ('N_0mmmmmm', 'N_0mmmmmmm', {'color': 'black'}),
        ('N_0mmmmmmm', 'N_0mmmmmmmm', {'color': 'black'}),
        ('N_0mmmmmmmm', 'N_0mmmmmmmmm', {'color': 'black'}),
        ('N_0mmmmmmmmm', 'N_0mmmmmmmmmm', {'color': 'black'}),
        ('N_0mmmmmmmmmm', 'N_0mmmmmmmmmmm', {'color': 'black'}),

        ('Root', 'N_1mmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmm', 'N_1mmmmmmmm', {'color': 'black'}),
        ('N_1mmmmmmmm', 'N_1mmmmmmmmm', {'color': 'black'}),

        ('N_0mmmmmmm', 'N_0mmmmmmmmmm', {'color': 'red', 'multiplicity': 1}),
        ('N_1mmmmmmmmm', 'N_0mmmmmmmmm', {'color': 'red', 'multiplicity': 1}),
        ('N_1mmmmmmmmm', 'N_0mmmmmmmmmmm', {'color': 'red', 'multiplicity': 1}),
    ])
    other_tree.bottom_node = 'N_0mmmmmmmmmmm'
    other_tree.current_level = 6

    print("\nBEFORE MERGE (2nd case):")
    self_tree.draw_tree(1)
    other_tree.draw_tree(2)

    self_tree.merge_trees(other_tree)

    print("\nAFTER MERGE (2nd case):")
    self_tree.draw_tree(2)

#test_merge_trees_second_case()