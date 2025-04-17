class GraphCollection :
    def __init__(self):
        self.G_prev = None
        self.G_curr = None
        self.HT_list_prev = None
        self.HT_list_curr = None

    def add_next_round(self, G, HT_list):
        self.G_prev = self.G_curr
        self.HT_list_prev = self.HT_list_curr
        self.G_curr = G
        self.HT_list_curr = HT_list
