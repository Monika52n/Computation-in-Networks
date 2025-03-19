from history_tree import HistoryTree

class Agent:
    def __init__(self, n, input_value):
        self.n = n  # Number of agents in the network
        self.input_value = input_value  # Agent's input
        self.myHT = HistoryTree(input_value)  # Current view of the history tree

    def input(self):
        return self.input_value

    def output(self, message):
        print(f"Agent output: {message}")

    def send_to_all_neighbors(self, message):
        # Simulate sending message to all neighbors (in a real case, this would
        # send the message to all connected agents in the network)
        return message

    def receive_from_all_neighbors(self, receivedMessages):
        self.receivedMessages = receivedMessages

    def chop(self, history_tree):
        # Chop level L0 from the history tree as described in the algorithm
        history_tree.chop()

    def compute_frequencies(self, history_tree):
        # Compute frequencies from the history tree
        return {}  # Dummy return, implement frequency computation logic

    def main(self):
        if self.myHT.get_max_height() > 2 * self.n - 2:
            self.myHT = HistoryTree(self.input_value)
        
        allMessages = self.receivedMessages + [self.myHT]
        minHT = min(allMessages, key=lambda ht: ht.get_max_height())

        while self.myHT.get_max_height() > minHT.get_max_height():
            self.chop(self.myHT)

        # Add a new bottom to the history tree
        self.myHT.add_bottom(self.input_value)

        for HT in self.receivedMessages:
            while HT.get_max_height() > minHT.get_max_height():
                self.chop(HT)

            # Match and merge HT into myHT
            self.myHT = self.merge_trees(self.myHT, HT)

            # Add a red edge (simulated)
            # ide kell valami ilyesmi: self.myHT.add_red_edge(HT.get_bottom(), self.myHT.get_bottom())

        if self.myHT.get_max_height() == 2 * self.n - 1:
            self.chop(self.myHT)

        if True: #"counting_level" in self.myHT:
            self.output(self.compute_frequencies(self.myHT))
        else:
            self.output([(self.input_value, 100)])

    def merge_trees(self, tree1, tree2):
        # This function will merge two history trees (simplified)
        return tree1  # Simplified, implement tree merging logic