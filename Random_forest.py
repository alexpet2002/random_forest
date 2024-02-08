import os
import glob
import random

import nltk
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

import id3

nltk.download('punkt')
from nltk.corpus import stopwords

# hyperparameters
# m = 100  # Top m most frequent words
# n = 50  # Skip the top n most frequent words
# k = 50  # Skip the top k rare words

# step1: load the text
train_directory_pos = 'C:/Users/alex/Desktop/aclImdb_v1/aclImdb/train/pos'
train_directory_neg = 'C:/Users/alex/Desktop/aclImdb_v1/aclImdb/train/neg'
test_directory_pos = 'C:/Users/alex/Desktop/aclImdb_v1/aclImdb/test/pos'
test_directory_neg = 'C:/Users/alex/Desktop/aclImdb_v1/aclImdb/test/neg'


# step2: tokenize the text

def read_file(file_path):
    try:
        file_list = glob.glob(os.path.join(file_path, '*.txt'))
        for file_name in file_list:
            current_file_path = os.path.join(file_path, file_name)
            with open(current_file_path, 'r') as file:
                # Do something with the file, e.g., read its contents
                content = file.read()
                word_list = content.split()
                return word_list
                print(word_list)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_vocab(tokenized_text, m, n, k):
    # Flatten the tokenized texts
    all_words = [word for sublist in tokenized_text for word in sublist]

    # show and remove stopwords
    nltk.download('stopwords')
    print(stopwords.words('english'))
    stop_words = set(stopwords.words('english'))
    # filter the words so that the list doesn't include any stopwords
    filtered_words = [word for word in all_words if word.isalnum() and word not in stop_words]
    print(filtered_words)

    # nltk.FreqDist info:
    # Construct a new frequency distribution. If samples is given, then the frequency distribution will be initialized with the count of each object in samples; otherwise, it will be initialized to be empty.
    # In particular, FreqDist() returns an empty frequency distribution; and FreqDist(samples) first creates an empty frequency distribution, and then calls update with the list samples.
    word_freq = nltk.FreqDist(filtered_words)

    # find m,n,k most frequent words
    common_words_and_freq = word_freq.most_common(m + n + k)
    # remove the respective frequencies
    common_words = [element for element, _ in common_words_and_freq]

    # define the vocabulary
    # Create a binary vector for each text
    filtered_vocab = common_words[n:n + m]
    print(filtered_vocab)
    return filtered_vocab


def tokenize_text(data):
    # Tokenize the texts and make the words lowercase
    tokenized_texts = [word_tokenize(text.lower()) for text in data]
    return tokenized_texts


def text_to_array(data, filtered_vocab):
    binary_array = [1 if char in filtered_vocab else 0 for char in data]
    return binary_array


def vectorize_text(data, filtered_vocab):
    vectorized_text = CountVectorizer(vocabulary=filtered_vocab, binary=True)
    x = vectorized_text.fit_transform(data)

    # Convert counts to binary values
    result = x.toarray()
    result[result > 1] = 1

    print(result)
    return result


def create3d_matrix(category, vectorized_text):
    label = ("neg" if category == 0 else "pos")
    array_of_sets = []
    for element in vectorized_text:
        array_of_sets.append([element, label])
    print(array_of_sets)


def bootstrapping(text_files_path):
    # List all files in the folder
    files = os.listdir(text_files_path)

    # Choose a random file from either positive or negative category
    random_file = random.choice(files)
    return random_file


def add_label(folder_label, array):
    label = ("neg" if folder_label == 0 else "pos")
    return [array, label]


def select_random_samples(filepath):
    pass


# important!!! should be applied to pos and neg separately
def create_matrix_and_vocab(filepath, folder_label):
    text_collection = select_random_samples(filepath)
    test_data = read_file(filepath)
    tokenized_text = tokenize_text(test_data)
    test_vocabulary = create_vocab(tokenized_text, 5, 30, 5)
    matrix = []
    for i in range(len(text_collection)):
        vectorized_text = vectorize_text(text_collection[i], test_vocabulary)
        array = text_to_array(vectorized_text, vocabulary)
        matrix.append(add_label(folder_label, array))

    return matrix, test_vocabulary


def train_trees(filepath, folder_label, n):
    # n indicates number of trees
    initial_matrix = create_matrix_and_vocab(filepath, folder_label)[0]
    initial_vocabulary = create_matrix_and_vocab(filepath, folder_label)[1]
    trained_trees = []
    for i in range(n):
        new_vocabulary = break_vocabulary(initial_vocabulary, 4)
        new_matrix = break_matrix(initial_matrix, new_vocabulary)
        node = id3.Node(None, None, new_matrix, None, None, None,None, None)
        tree = id3.Tree(node)
        tree.construct_tree(node)
        trained_trees.append(tree)
    return trained_trees


def break_vocabulary(vocabulary, m):
    # m indicates the number of attributes
    # returns a random subset of a vocabulary
    pass


def break_matrix(intial_matrix, vocabulary):
    # returns a subset ov values based on vocabulary
    pass


def pick_a_tree(tree_collection):
    return random.choice(tree_collection)


def traverse_tree(tree_collection, data):
    while not (len(tree_collection) == 0):
        for element in data:
            selected_tree = pick_a_tree(tree_collection)
            tree_collection.remove(selected_tree)
            current_node = selected_tree.root
            if current_node.is_leaf_node():
                return current_node.final_decision()
            if element.contains_attribute(current_node.best_attribute):
                traverse_tree(tree_collection, current_node.left_side)
            else:
                traverse_tree(tree_collection, current_node.right_side)


def random_forest(train_folder_path, test_folder_path):
    # train data
    trained_trees = train_trees(train_folder_path)


if __name__ == '__main__':
    filepath = "C:/Users/alex/Desktop/txt_test"
    data = read_file(filepath)
    tokenized_text = tokenize_text(data)
    vocabulary = create_vocab(tokenized_text, 5, 30, 5)
    vectorized_text = vectorize_text(data, vocabulary)
    # print(text_to_array(data, vocabulary))
    create3d_matrix(0, vectorized_text)

    testa = [0, 0, 0, 0]
    testb = [1, 0, 0, 0]
    testac = [0, 0, 0, 1]

    testf = []
    testf.append(testa)
    testf.append(testb)
    testf.append(testac)
    # probab = calculate_probabilities(testf)
    # print(calculate_entropy(probab))
