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


class Node:
    ## root node: original table ( which is a random subset of the merged data)
    ## left node: filtered table for 1 (absence of word)  and negative outcome
    # the table should contain the info for all the other words too
    ## right node: filtered table for 0...
    # for each node the entropy and information gain should be stored

    def __init__(self, entropy=None, inf_gain=None, matrix=None, left_side=None, right_side=None, parent_node=None):
        self.entropy = entropy  # entropy at this node
        self.inf_gain = inf_gain  # information gain of this node
        self.matrix = matrix  # matrix to be examined
        self.left_side = left_side  # left node
        self.right_side = right_side  # right node
        self.parent_node = parent_node  # parent node

    def filter_data(self, word_presence, word):
        filtered_data = [element for element in self.matrix if element[word] == word_presence]
        return filtered_data

    def leaf_node(self):
        return all(i == self.matrix[i][1] for i in self.matrix)

    def calculate_probabilities(self):
        vocab_length = len(self.matrix[0][0])
        probability_matrix = [0] * vocab_length  # Initialize with a list of zeros
        matrix_size = len(self.matrix)

        for i in range(matrix_size):
            for j in range(vocab_length):
                probability_matrix[j] += self.matrix[i][0][j]

        for i in range(vocab_length):
            probability_matrix[i] /= matrix_size

        return probability_matrix

    def calculate_information_gain(self):
        # entropy for each attribute in the matrix
        probabilities = self.calculate_probabilities()
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
        for element in vocabulary:
            # for each attribute in the vocabulary, try and split the data into subsets and calculate the information
            # gain for each subset, return the best attribute
            self.split_data(1, element, self.entropy, self.inf_gain)
            # calculate information gain for each value of attribute (0 and 1)
            # TODO: code looks questionable, review it
            information_gain_1 = self.left_side.calculate_information_gain()[0]
            information_gain_0 = self.right_side.calculate_information_gain()[0]
            # sum of information gains
            sum_of_gains = information_gain_0 + information_gain_1
            if sum_of_gains > max_information_gain:
                max_information_gain = sum_of_gains
                best_attribute = element
        return best_attribute, max_information_gain

    def split_data(self, word_presence, word, entropy, inf_gain):
        left_side = self.filter_data(word_presence, word)
        right_side = self.filter_data(inverse_word_presence(word_presence), word)
        left_node = Node(entropy, inf_gain, left_side, None, None, self)
        right_node = Node(entropy, inf_gain, right_side, None, None, self)
        # TODO: remove these 2?????
        self.left_side = left_node
        self.right_side = right_node
        return left_node, right_node

    class Id3_tree:

        def __init__(self, depth=None, leaf_nodes=None):
            self.leaf_nodes = leaf_nodes
            self.depth = depth
            self.root = None
            self.leaf_nodes = []

        def construct_tree(self, node):
            if node.leaf_node():
                self.leaf_nodes.append(node)

            best_attribute = node.select_word()[0]
            info_gain_of_attribute = node.select_word()[1]
            node.split_data(1, best_attribute, node.entropy, info_gain_of_attribute)
            left_child_entropy = calculate_entropy(node.left_side.calculate_probabilities())
            right_child_entropy = calculate_entropy(node.left_side.calculate_probabilities())
            left_child = node.left_side
            right_child = node.right_side
            left_child.entropy = left_child_entropy
            right_child.entropy = right_child_entropy

            self.construct_tree(left_child)
            self.construct_tree(right_child)

            return self.leaf_nodes


if __name__ == '__main__':
    testa = [0,0,0,0]
    testb = [1,0,0,0]
    testac = [0,0,0,1]
    testf = []
    testf.append(testa)
    testf.append(testb)
    testf.append(testac)
    initial_node = Node(testf,)
