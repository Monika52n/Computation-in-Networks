import random
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

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

    """ def __init__(self, root_label):
        self.G = nx.MultiDiGraph()
        self.root = root_label
        self.G.add_nodes_from([
            (root_label, {'label': root_label, 'level': -1})
        ])
        self.G.graph['Root'] = root_label
        self.bottom_node = root_label
        self.current_level = -1
        self.red_edges = defaultdict(int) """

    def get_tree(self):
        return self.G

    # merge 
    def merge_trees(self, other_tree):
        node_map = {}
        this_root = self.G.graph['Root']
        other_root = other_tree.G.graph['Root']
        node_map[other_root] = this_root

        new_nodes = []

        for level in range(-1, min(self.current_level, other_tree.current_level) + 1):
            this_level_nodes = [n for n, attr in self.G.nodes(data=True) if attr['level'] == level]
            other_level_nodes = [n for n, attr in other_tree.G.nodes(data=True) if attr['level'] == level]

            for other_node in other_level_nodes:
                if other_node in node_map:
                    continue

                other_attrs = other_tree.G.nodes[other_node]
                other_parent = next(other_tree.G.predecessors(other_node), None)

                #uj
                other_path = other_tree.get_path_to_root(other_node)
                matched = False
                for this_node in this_level_nodes:
                    this_attrs = self.G.nodes[this_node]
                    this_parent = next(self.G.predecessors(this_node), None)

                    # Útvonal-ellenőrzés
                    this_path = self.get_path_to_root(this_node)

                    if this_path == other_path:
                        for this_out_edge in self.G.out_edges(this_node, data=True):
                            for other_out_edge in other_tree.G.out_edges(other_node, data=True):
                                this_target_path = self.get_path_to_root(this_out_edge[1])
                                other_target_path = other_tree.get_path_to_root(other_out_edge[1])

                                if this_target_path == other_target_path:
                                    if this_out_edge[2]['color'] == 'red' and other_out_edge[2]['color'] == 'red':
                                        self.red_edges[(this_node, this_out_edge[1])] += other_out_edge[2]['multiplicity']
                                        this_out_edge[2]['multiplicity'] += other_out_edge[2]['multiplicity']

                        for other_out_edge in other_tree.G.out_edges(other_node, data=True):
                            if other_out_edge[2]['color'] == 'red':
                                other_target_path = other_tree.get_path_to_root(other_out_edge[1])
                                for node in self.G.nodes():
                                    this_target_path = self.get_path_to_root(node)
                                    if this_target_path == other_target_path:
                                        self.red_edges[(this_node, node)] += 1
                                        self.G.add_edge(this_node, node, color='red', multiplicity=self.red_edges[(this_node, node)])


                        # print(f"Skipping {other_node}, because the path matches an existing node: {this_node}")
                        node_map[other_node] = this_node
                        matched = True
                        break

                if not matched:
                    # Ha nem találtunk egyező útvonalat, létrehozunk egy új csomópontot

                    if other_parent and other_parent in node_map:
                        parent_mapped = node_map[other_parent]

                        # Ellenőrizzük, hogy van-e már ilyen értékű gyermek
                        existing_children = list(self.G.successors(parent_mapped))
                        duplicate_found = any(self.G.nodes[child]['label'] == other_attrs['label'] for child in existing_children)



                        if duplicate_found:
                            print(f"Skipping {other_node}, because an identical labeled child already exists under {parent_mapped}")
                            node_map[other_node] = next(child for child in existing_children if self.G.nodes[child]['label'] == other_attrs['label'])
                            node_to_check_for_in_edge = parent_mapped
                        else:
                            new_node = f"{other_node}_m"
                            print(f"Creating new node {new_node} for {other_node}")

                            if parent_mapped != new_node:
                                self.G.add_node(new_node, label=other_attrs['label'], level=other_attrs['level'])
                                node_map[other_node] = new_node

                                print(f"Adding edge from {parent_mapped} to {new_node}")
                                self.G.add_edge(parent_mapped, new_node, color='black')

                                new_nodes.append(new_node)
                                node_to_check_for_in_edge = new_node
                            else:
                                print(f"WARNING: Self-loop detected and avoided: {parent_mapped} -> {new_node}")


                        for other_out_edge in other_tree.G.in_edges(other_node, data=True):
                            if other_out_edge[2]['color'] == 'red':
                                other_source_path = other_tree.get_path_to_root(other_out_edge[0])
                                for node in self.G.nodes():
                                    this_source_path = self.get_path_to_root(node)
                                    if this_source_path == other_source_path:
                                        self.red_edges[(node, node_to_check_for_in_edge)] += 1
                                        self.G.add_edge(node, node_to_check_for_in_edge, color = 'red', multiplicity=self.red_edges[(node, node_to_check_for_in_edge)])


        if other_tree.bottom_node in node_map:
            mapped_bottom = node_map[other_tree.bottom_node]
            new_bottom_node = f"{self.bottom_node}_m"
            this_bottom_node = self.G.nodes[self.bottom_node]

            #Add new node that is the same as old bottom node, and is the child of the old bottom node (black edge)
            self.G.add_nodes_from([
                (new_bottom_node, {'label': this_bottom_node['label'], 'level': this_bottom_node['level'] + 1})
            ])
            self.G.add_edge(self.bottom_node, new_bottom_node, color='black')

            #predecessors = list(self.G.predecessors(new_bottom_node))
            #if predecessors[0] != mapped_bottom:
            self.red_edges[(mapped_bottom, new_bottom_node)] += 1
            self.G.add_edge(mapped_bottom, new_bottom_node, color='red', multiplicity=self.red_edges[(mapped_bottom, new_bottom_node)])
            self.bottom_node = new_bottom_node
            '''else:
                self.G.remove_node(new_bottom_node)'''


        return node_map

    def get_path_to_root(graph, node):
        # Visszaadja a csomópont útvonalát a gyökérig (ancestor chain). 
        path = []
        # print('get_path_to_root: node: ', node)
        # print('nodes before ERR: ', graph.G.nodes)
        while node is not None:
            path.append(graph.G.nodes[node]['label'])  # Az útvonalban a címkét tároljuk
            predecessors = list(graph.G.predecessors(node))
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
    def chop(self):
        if len(self.get_path_to_root(self.bottom_node)) > 2:
            """ print("\n--- CHOP START ---")
            print("Tree before chop:")
            print("Nodes:", list(self.G.nodes(data=True)))
            print("Edges:", list(self.G.edges(data=True))) """

            if not self.G.nodes():
                return

            # Step 1: Identify L0 nodes (direct children of root)
            l0_nodes = list(self.G.successors("Root"))
            
            # If there are no L0 nodes, nothing to chop
            if not l0_nodes:
                return

            # Step 2: Collect edges to preserve and nodes to update
            edges_to_preserve = {
                'black': defaultdict(int),
                'red': defaultdict(int)
            }
            nodes_to_update = {}
            nodes_to_keep = set()

            for l0_node in l0_nodes:
                # Collect all edges from L0 nodes
                for _, neighbor, data in self.G.out_edges(l0_node, data=True):
                    edge_type = data.get('color', 'black')
                    if edge_type == 'black':
                        edges_to_preserve_key = ("Root", neighbor)
                    else:
                        edges_to_preserve_key = (l0_node, neighbor)
                    
                    edges_to_preserve[edge_type][edges_to_preserve_key] += data.get('multiplicity', 1)
                    nodes_to_keep.add(neighbor)

                # Collect all nodes that need level updates
                for node in nx.dfs_preorder_nodes(self.G, source=l0_node):
                    current_level = self.G.nodes[node]['level']
                    nodes_to_update[node] = current_level - 1 if current_level > 0 else current_level

            # Step 3: Remove only L0 nodes that have children (to prevent complete tree deletion)
            l0_nodes_to_remove = [node for node in l0_nodes if any(self.G.out_edges(node))]
            self.G.remove_nodes_from(l0_nodes_to_remove)

            # Step 4: Update levels for remaining nodes
            for node, new_level in nodes_to_update.items():
                if node in self.G.nodes and node != "Root":
                    self.G.nodes[node]['level'] = new_level

            # Step 5: Restore all edges carefully
            for edge_type in ['black', 'red']:
                for (u, v), m in edges_to_preserve[edge_type].items():
                    if v in self.G.nodes and (u == "Root" or u in self.G.nodes):
                        if not self.G.has_edge(u, v):
                            self.G.add_edge(u, v, color=edge_type, multiplicity=m)
                        else:
                            # If edge already exists, just update multiplicity
                            self.G.edges[u, v]['multiplicity'] += m

            # Step 6: Merge isomorphic nodes
            while self._merge_all_levels():
                pass
                
            """ print("--- CHOP END ---")
            print("Tree after chop:")
            print("Nodes:", list(self.G.nodes(data=True)))
            print("Edges:", list(self.G.edges(data=True)))
            print("--- END ---\n") """
        

    def chop2(self):
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
            print("--- END ---\n")

    def _merge_nodes(self, representative, node):
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
        self.G.remove_node(node)

    def _merge_all_levels(self):
        """Merge nodes with identical sub-views at all levels in a single pass"""
        merged = False
        levels = sorted(set(data.get('level', -2) for _, data in self.G.nodes(data=True)))
        for level in levels:
            if level < 0:
                continue
            #self.draw_tree(2)
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
            #print(self._hash_sub_view(node))

        merged = False
        for group in groups.values():
            if len(group) > 1:
                representative = group[0]
                for node in group[1:]:
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

    '''def _hash_sub_view(self, node):
        """Create a hashable representation of the sub-view rooted at node"""
        sub_view = {
            'label': self.G.nodes[node].get('label', ''),
            'value'
            'parents': []
        }

        for p in sorted(self.G.predecessors(node)):
            edge_data = self.G.edges[p, node, 0]
            if edge_data.get('color') == 'black':
                sub_view['parents'].append(('black', self.G.nodes[p].get('label', '')))

        sub_view['parents'].sort()
        print('sub', sub_view)
        return str(sub_view)'''

    def add_bottom(self, input_value):
        print('input_value: ', input_value)
        print('bottom: ', self.bottom_node)
        print('nodes: ', self.G.nodes)

        if len(self.G.nodes) == 1:
            self.__init__('Root', input_value)
        else:
            this_node = self.G.nodes[self.bottom_node]
            new_label = f"{self.bottom_node}_m"

            self.G.add_nodes_from([
                (new_label, {'label': input_value, 'level': this_node['level'] + 1})
            ])
            self.G.add_edge(self.bottom_node, new_label, color='black')
            self.bottom_node = new_label

    def get_bottom(self):
        pass

    def compute_frequencies(self):
        # Itt pl. egy egyszerű számítás lehetne, hogy hány "Input" van a fában
        input_count = sum(1 for node, attr in self.G.nodes(data=True) if attr['label'] == 'Input')
        return input_count

    def get_max_height(self):
        return nx.dag_longest_path_length(self.G, 'Root') if self.G.nodes else 0


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
        ("E", "G", {'color': 'black'})
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
        ("L", "N", {'color': 'black'})
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

    print("Before chop:")
    ht2.draw_tree()
    
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
    ht2.draw_tree()

# Run the test
#test_ht2_chop()

def test_linear_tree_chop():
    # Initialize history tree with root
    tree = HistoryTree("Root")
    
    # Build the specified tree structure
    tree.G.add_nodes_from([
        ('Root', {'label': 'Root', 'level': -1}),
        ('a0', {'label': 'a', 'level': 0}),
        ('b0', {'label': 'b', 'level': 0}),
        ('c0', {'label': 'c', 'level': 0}),
        ('a1', {'label': 'a', 'level': 1}),
        ('b1', {'label': 'b', 'level': 1}),
        ('b1_2', {'label': 'b', 'level': 1}),
        ('a2', {'label': 'a', 'level': 2}),
        ('b2', {'label': 'b', 'level': 2}),
        ('b2_2', {'label': 'b', 'level': 2}),
        ('a3', {'label': 'a', 'level': 3})
    ])
    
    # Add black edges
    tree.G.add_edges_from([
        ("Root", "a0", {'color': 'black'}),
        ("Root", "b0", {'color': 'black'}),
        ("Root", "c0", {'color': 'black'}),
        ("a0", "a1", {'color': 'black'}),
        ("b0", "b1", {'color': 'black'}),
        ("c0", "b1_2", {'color': 'black'}),
        ("a1", "a2", {'color': 'black'}),
        ("b1", "b2", {'color': 'black'}),
        ("b1_2", "b2_2", {'color': 'black'}),
        ("a2", "a3", {'color': 'black'})
    ])
    
    # Add some red edges for testing
    tree.G.add_edges_from([
        ("a0", "b1", {'color': 'red', 'multiplicity': 2}),
        ("b0", "a1", {'color': 'red', 'multiplicity': 1}),
        ("c0", "b2", {'color': 'red', 'multiplicity': 1})
    ])
    
    tree.bottom_nodes = ['a3']
    tree.current_level = 3

    print("Initial Tree Structure:")
    _safe_draw_tree(tree.G)  # Use safe visualization
    
    # Perform chop operation
    print("\nPerforming chop operation...")
    tree.chop()

    
    print("\nDetailed Node Information (After Chop):")
    for node, data in tree.G.nodes(data=True):
        print(f"Node {node}: label={data['label']}, level={data['level']}")
    
    print("\nDetailed Edge Information (After Chop):")
    for u, v, data in tree.G.edges(data=True):
        edge_type = "BLACK" if data.get('color') == 'black' else "red"
        mult = data.get('multiplicity', 1)
        print(f"{edge_type} edge {u} -> {v} (multiplicity={mult})")

    tree.draw_tree()
    # Verification
    print("\nVerification:")

    # Check level 0 nodes were removed
    assert not any(data['level'] == 0 and node not in ['a1', 'b1', 'b1_2'] 
                for node, data in tree.G.nodes(data=True)), "Old level 0 nodes were not removed"

    # Check former level 1 nodes are now level 0
    for node in ['a1', 'b1', 'b1_2']:
        if node in tree.G.nodes():
            assert tree.G.nodes[node]['level'] == 0, f"Node {node} level update failed"

    # Check level consistency
    for node, data in tree.G.nodes(data=True):
        if node == 'Root':
            continue
        parents = list(tree.G.predecessors(node))
        if parents:
            parent_level = tree.G.nodes[parents[0]]['level']
            assert data['level'] == parent_level + 1, \
                f"Level mismatch at {node} (has {data['level']}, parent has {parent_level})"

    # Check red edges were preserved
    red_edges = [(u, v, d['multiplicity']) for u, v, d in tree.G.edges(data=True) 
                if d.get('color') == 'red']
    assert len(red_edges) >= 2, f"Expected at least 2 red edges, found {len(red_edges)}"
    print("Preserved red edges:")
    for u, v, m in red_edges:
        print(f"  {u} -> {v} (multiplicity={m})")
    
    print("\nAll tests passed!")

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

    ht2.chop()

    ht2.draw_tree(2)


# Run the test
#test_chop()


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
        ("B1", "C1", {'color': 'red', 'multiplicity': 1}),
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
    ht2.merge_trees(ht1)
    ht2.draw_tree(2)

#test_merge_trees2()





