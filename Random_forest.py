import copy
import os
import glob
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import id3

nltk.download('punkt')
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score

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

def read_and_merge_files(file_path):
    try:
        file_list = glob.glob(os.path.join(file_path, '*.txt'))
        for file_name in file_list:
            current_file_path = os.path.join(file_path, file_name)
            with open(current_file_path, 'r') as file:
                # Do something with the file, e.g., read its contents
                content = file.read()
                word_list = content.split()
                return word_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_single_files(file_path):
    words = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content of the file
            content = file.read()

            # Split the content into words using whitespace as a separator
            words = content.split()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return words


def get_file_directories(folder_path):
    file_array = []
    # Check if the folder path exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Get a list of all files in the folder
        files = os.listdir(folder_path)

        # Iterate through each file in the folder
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            # Check if the path is a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Append the file path to the array
                file_array.append(file_path)

    # Print the array of file paths
    print(file_array)
    return file_array


def create_vocab(tokenized_text, m, n, k):
    # Flatten the tokenized texts
    all_words = [word for sublist in tokenized_text for word in sublist]

    # show and remove stopwords
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    # filter the words so that the list doesn't include any stopwords
    filtered_words = [word for word in all_words if word.isalnum() and word not in stop_words]

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
    # print(result)
    return result


def create3d_matrix(category, vectorized_text):
    label = ("neg" if category == 0 else "pos")
    array_of_sets = []
    for element in vectorized_text:
        array_of_sets.append([element, label])
    print(array_of_sets)


def extract_data_from_files(text_files_path):
    # List all files in the folder
    files = os.listdir(text_files_path)

    # Choose a random file from either positive or negative category
    random_file = random.choice(files)
    return random_file


def bootstrap_data(matrix, vocabulary, num_of_attributes=3):
    bootstrap_vocab = np.random.choice(vocabulary, size=num_of_attributes, replace=False)
    bootstrap_vocab_indx = [id3.find_index_from_word(element, vocabulary) for element in bootstrap_vocab if
                            element in vocabulary]

    bootstrap_indx = random_indices_with_repetition(len(matrix) - 1, len(matrix))

    bootstrapped_rows = [matrix[bootstrap_indx[i]] for i in range(len(bootstrap_indx))]
    bootstrapped_rows_copy = copy.deepcopy(bootstrapped_rows)
    bootstrapped_matrix = filter_matrix(bootstrap_vocab_indx, bootstrapped_rows_copy)
    return bootstrapped_matrix, bootstrap_vocab


def filter_matrix(bootstrap_vocab_indx, bootstrapped_rows):
    filtered_matrix = []
    for j in range(len(bootstrapped_rows)):
        array = bootstrapped_rows[j][0]
        new_array = filter_array_by_indices(array, bootstrap_vocab_indx)
        filtered_matrix.append([new_array, bootstrapped_rows[j][1]])
    return filtered_matrix


def filter_array_by_indices(input_array, indices):
    filtered_array = [input_array[element] for element in indices]
    return filtered_array


def add_label(folder_label, array):
    label = ("neg" if folder_label == 0 else "pos")
    return [array, label]


def create_matrix(filepath, folder_label, vocabulary):
    directory = get_file_directories(filepath)
    data = []
    for element in directory:
        words = read_text_file(element)
        data.append(words[0])
    vectorized_text = vectorize_text(data, vocabulary)
    matrix = []
    for element in vectorized_text:
        matrix.append(add_label(folder_label, element))

    return matrix


def merge(positive, negative):
    return positive + negative


def train_trees(pos_directory, neg_directory, n):
    # n indicates number of trees
    test_data_1 = read_and_merge_files(pos_directory)
    test_data_2 = read_and_merge_files(neg_directory)
    tokenized_text = tokenize_text(test_data_1) + tokenize_text(test_data_2)
    initial_vocabulary = create_vocab(tokenized_text, 5, 30, 5)
    print("vocabulary to be used: ")
    print(initial_vocabulary)
    pos_initial_matrix = create_matrix(pos_directory, "pos", initial_vocabulary)
    neg_initial_matrix = create_matrix(neg_directory, "neg", initial_vocabulary)
    initial_matrix = merge(pos_initial_matrix, neg_initial_matrix)
    print("matrix to be used:")
    print(initial_matrix)
    trained_trees = []
    for i in range(n):
        new_matrix_new_vocab = bootstrap_data(initial_matrix, initial_vocabulary)
        new_matrix = new_matrix_new_vocab[0]
        new_vocab = new_matrix_new_vocab[1]
        node = id3.Node(None, None, new_matrix, None, None, None, new_vocab, None)
        tree = id3.Tree(node)
        tree.construct_tree(node)
        print(tree.root.best_attribute)
        trained_trees.append(tree)
    return trained_trees


def pick_a_tree(tree_collection):
    return random.choice(tree_collection)


def read_text_file(file_path):
    text_array = []

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read each line and append it to the array
            for line in file:
                text_array.append(line.strip())  # strip() removes leading and trailing whitespaces

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return text_array


def traverse_tree(tree, data, node):
    if node is None:
        node = tree.root
    best_attribute = node.best_attribute
    attribute_index = id3.find_index_from_word(best_attribute, node.vocabulary)
    if node.is_leaf_node():
        return node.final_decision()
    if data[attribute_index] == 1:
        traverse_tree(tree, node.left_side)
    else:
        traverse_tree(tree, node.right_side)


def random_forest(pos, neg, test_folder_path_pos, test_folder_path_neg, n):
    # train data
    print("Starting random forest algorithm!")
    print("Training trees...")
    trained_trees = train_trees(pos, neg, n)
    vocab = trained_trees[0].root.vocabulary
    test_data_directories = get_file_directories(test_folder_path_pos) + get_file_directories(test_folder_path_neg)
    predicted_labels = []
    for file_directory in test_data_directories:
        text = read_text_file(file_directory)
        vector_text = vectorize_text(text, vocab)
        prediction = final_decision_for_data(vector_text, trained_trees)
        print(f"Final decision is for the file: {file_directory} is {prediction}")
        predicted_labels.append(prediction)
    return test_data_directories, predicted_labels


def final_decision_for_data(test_text, trained_trees):
    predictions = []
    for tree in trained_trees:
        decision = traverse_tree(tree, test_text, tree.root)
        predictions.append(decision)
    count = Counter(predictions)
    return count.most_common(1)[0][0]


def random_indices_with_repetition(range_size, num_indices):
    return [random.randint(0, range_size - 1) for _ in range(num_indices)]


def evaluate_performance(test_data):
    true_labels = []

    for file_directory in test_data:
        true_label = "pos" if "pos" in file_directory else "neg"
        true_labels.append(true_label)
    return true_labels


def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="pos")
    f1 = f1_score(true_labels, predicted_labels, pos_label="pos")
    return accuracy, precision, f1


if __name__ == '__main__':
    train_negative_path = "C:/Users/alex/Desktop/aclImdb_v1/train/neg"
    train_positive_path = "C:/Users/alex/Desktop/aclImdb_v1/train/pos"
    test_positive_path = "C:/Users/alex/Desktop/aclImdb_v1/test/pos"
    test_negative_path = "C:/Users/alex/Desktop/aclImdb_v1/test/neg"
    returns = random_forest(train_positive_path, train_negative_path, test_positive_path, test_negative_path, 500)
    test_data_directories = returns[0]
    predicted_labels = returns[1]
    true_labels = evaluate_performance(test_data_directories)
    accuracy, precision, f1 = calculate_metrics(true_labels, predicted_labels)

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
