from agent import Agent
from history_tree import HistoryTree

# Simulating the agents in the network
def run_simulation(n, agents_inputs):
    agents = [Agent(n, input_value) for input_value in agents_inputs]
    for agent in agents:
        agent.send_to_all_neighbors("message")
    for agent in agents:
        agent.receive_from_all_neighbors([HistoryTree(agent.input())])
    for agent in agents:
        agent.main()


# Example: running the simulation with 3 agents
agents_inputs = [1, 0, 0]  # Example input for 3 agents
n = len(agents_inputs)
run_simulation(n, agents_inputs)
