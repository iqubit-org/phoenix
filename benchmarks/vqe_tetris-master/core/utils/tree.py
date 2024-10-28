import queue
import pdb

class TreeNode:
    def __init__(self, idx, parent):
        self.idx = idx
        self.parent = parent
        self.idx_after_swap = idx
        self.parent_after_swap = parent

class Tree:
    def __init__(self, edges, root):
        self.node_list = []
        self.root = root
        self.flood_fill(root, edges)
        # the node_list have all nodes in the order of distance from root from low to high
        # reverse the node_list so it's convenient to run the Pauli String
        self.node_list.reverse()
    
    def flood_fill(self, root, edges):
        visited = [root]
        self.node_list.append(TreeNode(root, -1))
        Q = queue.Queue()
        Q.put(root)
        while not Q.empty():
            node = Q.get()
            for edge in edges:
                if node in edge:
                    adj = edge[0] + edge[1] - node
                    if not adj in visited:
                        visited.append(adj)
                        self.node_list.append(TreeNode(adj, node))
                        Q.put(adj)

    def refresh(self):
        for node in self.node_list:
            node.idx_after_swap = node.idx
            node.parent_after_swap = node.parent
    
    def get_node(self, idx_after_swap):
        for node in self.node_list:
            if node.idx_after_swap == idx_after_swap:
                return node
        return None
    
    def exchange(self, child, parent):
        child.idx_after_swap, parent.idx_after_swap = parent.idx_after_swap, child.idx_after_swap
    
    def swap_two_nodes(self, parent, child):
        node_child = self.get_node(child)
        node_parent = self.get_node(parent)
        self.exchange(node_child, node_parent)
        for node in self.node_list:
            if node.parent_after_swap == parent:
                node.parent_after_swap = child