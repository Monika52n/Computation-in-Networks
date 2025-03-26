from anytree import Node, RenderTree
from anytree.exporter import DotExporter

root = Node("Root")
child1 = Node("Child1", parent=root)
child2 = Node("Child2", parent=root)
leaf = Node("Leaf", parent=child1)

for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name}")

DotExporter(root).to_picture("tree.png")  # PNG-be menti
