from collections import deque

def create_tree(leaf_list):
    if len(leaf_list) == 1:
        return leaf_list[0]
    else:
        new_node_list = deque()
        while leaf_list:
            try:
                l1 = leaf_list.popleft()
                l2 = leaf_list.popleft()
            except IndexError:
                l2 = None
            new_node_list.append(Node(l1, l2))
        return create_tree(new_node_list)


class Node:
    value = 0

    def __init__(self, lchild, rchild=None, parent=None):
        self.parent = parent
        self.left_child = lchild
        self.right_child = rchild if rchild else Leaf()

        self.update_value()

        self.left_child.parent = self
        self.right_child.parent = self

    def is_root(self):
        return self.parent is None

    def update_value(self):
        self.value = self.right_child.value + self.left_child.value


class Leaf(Node):
    idx = None
    left_child = None
    right_child = None

    def __init__(self, parent=None, value=0):
        self.parent = parent
        self.value = value

    def update(self, idx, value):
        self.idx = idx
        self.value = value


class SumTree:
    def __init__(self, n_leaf):
        self.leaf_stack = deque([Leaf() for _ in range(n_leaf)])
        self.root = create_tree(self.leaf_stack.copy())

    def get_sum(self):
        return self.root.value

    def add(self, idx, value):
        leaf = self.leaf_stack.popleft()
        leaf.update(idx, value)
        self.leaf_stack.append(leaf)
        self.propagate(leaf)
        return leaf

    def propagate(self, node):
        if node.is_root():
            return
        else:
            node.parent.update_value()
            self.propagate(node.parent)

    def get_leaf(self):
        def aux(node):
            if isinstance(node, Leaf):
                return [node]
            else:
                return aux(node.right_child) + aux(node.left_child)

        return deque(aux(self.root))


if __name__ == '__main__':
    b = SumTree(3)
    b.add(5)
    for l in b.leaf_stack:
        print(l.value)

    b.add(3)
    for l in b.leaf_stack:
        print(l.value)

    b.add(4)
    for l in b.leaf_stack:
        print(l.value)
    print(b.get_sum())
    print(b.add(2))
    for l in b.leaf_stack:
        print(l.value)
    print(b.get_sum())
