import numpy as np


# helper functions
def calculate_entropy(probability_matrix):
    entropies = []
    for i in range(len(probability_matrix)):
        probability = probability_matrix[i]
        entropies.append(-probability * np.log2(probability + 1e-10) if probability != 0 else 0)
    return entropies


def calculate_total_entropy(entropy_matrix):
    return sum(entropy_matrix)


def inverse_word_presence(word_presence):
    new_word_presence = (0 if word_presence == 1 else 1)
    return new_word_presence


def calculate_probabilities(matrix):
    try:
        vocab_length = len(matrix[0][0])
    except Exception as e:
        return 0
    print(vocab_length)
    probability_matrix = [0] * vocab_length  # Initialize with a list of zeros
    matrix_size = len(matrix)
    try:
        for i in range(matrix_size):
            for j in range(vocab_length):
                probability_matrix[j] += matrix[i][0][j]

        for i in range(vocab_length):
            probability_matrix[i] /= matrix_size

    except Exception as e:
        print("the matrix was empty")
    return probability_matrix


def find_index_from_word(word, vocabulary):
    return vocabulary.index(word)


def find_word_from_index(index, vocabulary):
    return vocabulary[index]


class Node:
    ## root node: original table ( which is a random subset of the merged data)
    ## left node: filtered table for 1 (absence of word)  and negative outcome
    # the table should contain the info for all the other words too
    ## right node: filtered table for 0...
    # for each node the entropy and information gain should be stored

    def __init__(self, entropy=None, inf_gain=None, matrix=None, left_side=None, right_side=None, parent_node=None,
                 vocabulary=None):
        self.vocabulary = vocabulary
        self.entropy = entropy  # entropy at this node
        self.inf_gain = inf_gain  # information gain of this node
        self.matrix = matrix  # matrix to be examined
        self.left_side = left_side  # left node
        self.right_side = right_side  # right node
        self.parent_node = parent_node  # parent node

    def filter_data(self, word_presence, word_index):
        filtered_data = [element for element in self.matrix if element[0][word_index] == word_presence]
        return filtered_data

    def leaf_node(self):
        all(entry[1] == 'neg' for entry in self.matrix)

    def calculate_information_gain(self):
        # entropy for each attribute in the matrix
        probabilities = calculate_probabilities(self.matrix)
        entropies = calculate_entropy(probabilities)
        total_entropy = calculate_total_entropy(entropies)
        # set the total entropy for the current node
        self.entropy = total_entropy
        weighted_entropies = []
        for value in entropies:
            # Calculate the weight (proportion) of the subset
            value_weight = len(self.matrix) / len(self.parent_node.matrix)

            # Multiply the subset entropy by its weight
            weighted_entropy = value_weight * value

            weighted_entropies.append(weighted_entropy)

        # Calculate information gain
        information_gain = total_entropy - np.sum(weighted_entropies)

        return information_gain, total_entropy

    def select_word(self):
        vocabulary = self.matrix[0][0]
        best_attribute = None
        max_information_gain = -1
        for i in range(len(vocabulary)):
            # for each attribute in the vocabulary, try and split the data into subsets and calculate the information
            # gain for each subset, return the best attribute
            self.split_data(1, i, self.entropy, self.inf_gain)
            # calculate information gain for each value of attribute (0 and 1)
            # TODO: code looks questionable, review it
            information_gain_1 = self.left_side.calculate_information_gain()[0]
            information_gain_0 = self.right_side.calculate_information_gain()[0]
            # sum of information gains
            sum_of_gains = information_gain_0 + information_gain_1
            if sum_of_gains > max_information_gain:
                max_information_gain = sum_of_gains
                best_attribute = find_word_from_index(i, self.vocabulary)
        return best_attribute, max_information_gain

    def split_data(self, word_presence, word_index, entropy, inf_gain):
        left_side = self.filter_data(word_presence, word_index)
        right_side = self.filter_data(inverse_word_presence(word_presence), word_index)
        left_node = Node(entropy, inf_gain, left_side, None, None, self)
        right_node = Node(entropy, inf_gain, right_side, None, None, self)
        self.left_side = left_node
        self.right_side = right_node
        return left_node, right_node


class Tree:

    def __init__(self, root):
        self.root = root
        self.leaf_nodes = []

    def construct_tree(self, node):
        if node.leaf_node():
            self.leaf_nodes.append(node)

        best_attribute = node.select_word()[0]
        info_gain_of_attribute = node.select_word()[1]
        node.split_data(1, best_attribute, node.entropy, info_gain_of_attribute)
        left_child_entropy = calculate_entropy(calculate_probabilities(node.left_side.matrix))
        right_child_entropy = calculate_entropy(calculate_probabilities(node.right_side.matrix))
        left_child = node.left_side
        right_child = node.right_side
        left_child.entropy = left_child_entropy
        right_child.entropy = right_child_entropy

        self.construct_tree(left_child)
        self.construct_tree(right_child)

        return self.leaf_nodes


if __name__ == '__main__':
    testa = [0, 0, 0, 0]
    testb = [1, 0, 0, 0]
    testac = [1, 0, 0, 1]
    test_vocabulary = ["hun", "love", "boo", "bae"]
    test_matrix = []
    test_matrix.append([testa, "neg"])
    test_matrix.append([testb, "pos"])
    test_matrix.append([testac, "neg"])
    print(test_matrix)
    test_entropy = calculate_entropy(calculate_probabilities(test_matrix))
    print(test_entropy)
    initial_node = Node(test_entropy, 0, test_matrix, None, None, None, test_vocabulary)
    print(initial_node)
    test_tree = Tree(initial_node)
    test_tree.construct_tree(initial_node)
    print(test_tree.leaf_nodes)
