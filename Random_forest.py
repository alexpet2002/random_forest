import os
import glob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

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
    # TODO: remove the elements till the n index and anything later than m+n index
    # Create a binary vector for each text
    filtered_vocab = common_words[n:n + m]
    print(filtered_vocab)
    return filtered_vocab


def tokenize_text(data):
    # Tokenize the texts and make the words lowercase
    tokenized_texts = [word_tokenize(text.lower()) for text in data]
    return tokenized_texts


#
# def text_to_array(data, filtered_vocab):
#     binary_array = []
#     x = len(data)
#     for i in range(x):
#         if data[i] in filtered_vocab:
#             binary_array.append(1)
#         else:
#             binary_array.append(0)
#     return binary_array

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
    label = ("negative" if category == 0 else "positive")
    array_of_sets = []
    for element in vectorized_text:
        array_of_sets.append([element, label])
    print(array_of_sets)


if __name__ == '__main__':
    filepath = "C:/Users/alex/Desktop/txt_test"
    data = read_file(filepath)
    tokenized_text = tokenize_text(data)
    vocabulary = create_vocab(tokenized_text, 5, 30, 5)
    vectorized_text = vectorize_text(data, vocabulary)
    # print(text_to_array(data, vocabulary))
    create3d_matrix(0, vectorized_text)

    testa = [0,0,0,0]
    testb = [1,0,0,0]
    testac = [0,0,0,1]

    testf = []
    testf.append(testa)
    testf.append(testb)
    testf.append(testac)
    # probab = calculate_probabilities(testf)
    # print(calculate_entropy(probab))