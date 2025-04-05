from history_tree import HistoryTree
from collections import defaultdict

class Agent:

    def __init__(self, n, input_value):
        self.n = n  # Number of agents in the network
        self.input_value = input_value  # Agent's input
        self.receivedMessages = []
        self.myHT = HistoryTree('Root', self.input_value)
        self.myHT_new = self.myHT

    def input(self):
        return self.input_value

    def output(self, message):
        print(f"Agent output: {message}")

    def send_to_neighbor(self):
        return self.myHT

    def receive_from_neighbor(self, receivedMessage):
        self.receivedMessages.append(receivedMessage)

    def chop(self, history_tree):
        # Chop level L0 from the history tree as described in the algorithm
        history_tree.chop()

    def compute_frequencies(self, history_tree):
        """
        Kiszámítja az egyes csúcsok címkéinek relatív előfordulási gyakoriságát a történeti fában.

        Paraméter:
            history_tree: A HistoryTree objektum, amely tartalmazza a gráfot.

        Visszatérési érték:
            Egy szótár, ahol a kulcsok a címkék, az értékek pedig a relatív gyakoriságok.
        """
        from collections import defaultdict

        label_counts = defaultdict(int)
        total_nodes = 0

        # Címkék megszámlálása
        for node in history_tree.G.nodes:
            label = history_tree.G.nodes[node].get('label', None)
            if label is not None:
                label_counts[label] += 1
                total_nodes += 1

        # Ha nincs csomópont, térjünk vissza üres szótárral
        if total_nodes == 0:
            return {}

        # Relatív gyakoriságok kiszámítása
        result = {label: count / total_nodes for label, count in label_counts.items()}

        return result

    def update_ht(self):
        self.myHT = self.myHT_new

    def main(self, neighbors):
        print(f"Length {len(self.receivedMessages)}")
        if self.myHT.get_max_height() > 2 * self.n - 2:
            self.myHT = HistoryTree('Root', self.input_value)

        for neighbor in neighbors:
            #neighbor.receive_from_neighbor(self.send_to_neighbor()) #sending current ht to all neighbors
            self.receive_from_neighbor(neighbor.send_to_neighbor()) #receiving ht from all neighbors
        
        allMessages = self.receivedMessages + [self.myHT]
        minHT = min(allMessages, key=lambda ht: ht.get_max_height())

        while self.myHT_new.get_max_height() > minHT.get_max_height():
            self.chop(self.myHT_new)

        # Add a new bottom to the history tree
        self.myHT_new.add_bottom(self.input_value)

        for HT in self.receivedMessages:
            while HT.get_max_height() > minHT.get_max_height():
                self.chop(HT)
            self.myHT_new.merge_trees(HT)

        if self.myHT_new.get_max_height() == 2 * self.n - 1:
            self.chop(self.myHT_new)

        self.myHT_new.draw_tree()

        self.receivedMessages = []
        if True: #"counting_level" in self.myHT:
            self.output(self.compute_frequencies(self.myHT_new))
        else:
            self.output([(self.input_value, 100)])

    '''def merge_trees(self, tree1, tree2):
        # This function will merge two history trees (simplified)
        return tree1  # Simplified, implement tree merging logic'''


def test_compute_frequencies():
    ht = HistoryTree("Root")

    ht.G.add_nodes_from([
        ("A", {"label": "0", "level": 0}),
        ("B", {"label": "1", "level": 0}),
        ("C", {"label": "1", "level": 1}),
        ("D", {"label": "0", "level": 1}),
        ("E", {"label": "1", "level": 2}),
    ])

    ht.G.add_edges_from([
        ("Root", "A"), 
        ("Root", "B"), 
        ("A", "C"), 
        ("A", "D"), 
        ("C", "E")
    ])

    agent = Agent(n=5, input_value="Root")
    frequencies = agent.compute_frequencies(ht)
    print(frequencies)


# Teszt futtatása
#test_compute_frequencies()
