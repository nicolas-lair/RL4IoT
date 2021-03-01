class Node:
    def __init__(self, name, node_type, children=None, node_embedding=None):
        self.name = name
        self.node_type = node_type

        # self.father = father
        self.children = children

        self.node_embedding = node_embedding
        self.embedding_size = None

    def get_node_embedding(self):
        return self.node_embedding

    # def get_father_node(self):
    #     return self.father

    def get_children_nodes(self):
        if self.children is not None:
            return self.children
        else:
            raise NotImplementedError


class DescriptionNode(Node):
    def __init__(self, name, description, children, node_type='description_node'):
        super().__init__(name=name, children=children, node_type=node_type)
        self.description = description
        self.has_description = True

    def embed_node_description(self, embedder):
        self.node_embedding = embedder(self.description)
        self.embedding_size = max(self.node_embedding.shape)


class NoDescriptionNode(Node):
    def __init__(self, name, node_type, node_embedding, children):
        super().__init__(name=name, children=children, node_type=node_type)
        self.description = None
        self.has_description = False
        self.node_embedding = node_embedding
        self.embedding_size = max(self.node_embedding.shape)


if __name__ == "__main__":
    n = Node()
