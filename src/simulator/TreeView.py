

class Node:
    def __init__(self, name, description, children=None, node_embedding=None):
        self.name = name

        self.description = description
        self.node_embedding = node_embedding

        # self.father = father
        self.children = children
        self.has_description = description is not None

    def get_node_embedding(self):
        return self.node_embedding

    # def get_father_node(self):
    #     return self.father

    def get_children_nodes(self):
        if self.children is not None:
            return self.children
        else:
            raise NotImplementedError

if __name__ == "__main__":
    n = Node()