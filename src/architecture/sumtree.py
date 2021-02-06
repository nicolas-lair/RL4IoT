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


"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable


class SegmentTree:
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.
        Args:
            capacity (int)
            operation (function)
            init_value (float)
        """
        assert (
                capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
            self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int):
        """Initialization.
        Args:
            capacity (int)
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int):
        """Initialization.
        Args:
            capacity (int)
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


if __name__ == '__main__':
    a = SumSegmentTree(4)
    b = SumTree(4)

    b.add(0, 5)
    a[0] = 5
    for l in b.leaf_stack:
        print(l.value)
        print

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
