from collections import Counter, defaultdict
from machine_learning import split_data
import math, random, re, glob
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer


def tokenize(message):
    message = message.lower()  # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)  # extract the words
    radicals_words = porter_temmer(all_words)  # chama funcao para
    return set(radicals_words)  # remove duplicates


# Utilizar apenas radicais das palavras (pesquise por "Porter Stemmer");
def porter_temmer(words):
    ps = PorterStemmer()
    for w in words:
        new_words = ps.stem(w)
    return new_words


# Considerar não apenas presença de palavras, mas outras características
def special_caracter(words):
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    words = tokenizer.tokenize(words)
    return words


def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.items()]


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        # count spam and non-spam messages
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


# Analisar o conteúdo da mensagem e não apenas o Assunto;
def get_subject_data(path):
    data = []

    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")
    body_regex = re.compile(r"^$.*")

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn, 'r', encoding='ISO-8859-1') as file:
            body_detected = False
            body = ""
            for line in file:
                text = line
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))
                elif line.strip() == '' and not body_detected:
                    body_detected = True
                if body_detected:
                    body = body + line
            data.append((body, is_spam))

    return data


def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


def train_and_test_model(path):
    data = get_subject_data(path)
    random.seed(0)  # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
                  for subject, is_spam in test_data]

    counts = Counter((is_spam, spam_probability > 0.5)  # (actual, predicted)
                     for _, is_spam, spam_probability in classified)

    print(counts)

    classified.sort(key=lambda row: row[2])
    spammiest_hams = list(filter(lambda row: not row[1], classified))[-5:]
    hammiest_spams = list(filter(lambda row: row[1], classified))[:5]

    print("spammiest_hams", spammiest_hams)
    print("hammiest_spams", hammiest_spams)

    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_words = words[-5:]
    hammiest_words = words[:5]

    print("spammiest_words", spammiest_words)
    print("hammiest_words", hammiest_words)


if __name__ == "__main__":
    train_and_test_model(r"./emails/*/*")
