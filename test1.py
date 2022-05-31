# article: https://jaiprakashml.medium.com/article-spinning-with-python-1653438d4842

# Loading required Python packages
import nltk
import random
# Data url : used electronics/positive.review http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
# used for reading data of xml format
from bs4 import BeautifulSoup 

#Reading the data
reviews_path = r'positive.review'
soup = BeautifulSoup(open(reviews_path).read(), features="html.parser")
positive_reviews = soup.findAll('review_text')

trigrams = {}

# Extract trigrams from positive_reviews and insert into dictionary
# (w1, w3) is the key, [ w2 ] are the values

for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens)-2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# turn each array of middle-words into a probability vector

# Initialize an Empty Dictionary
trigram_probabilities = {} 
for k in trigrams:
    words = trigrams[k]
    # create a dictionary of word -> count
    if len(set(words)) > 1:
        # only do this when there are different possibilities for a middle word
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w in d:
            d[w] = float(d[w]) / n
        trigram_probabilities[k] = d

# choose a random sample from dictionary where values are the 
# probabilities
def random_sample(d):
    r = random.random() #Initialize an Empty list
    cumulative = 0
    for w in d:
        p = d[w]
        cumulative += p
        if r < cumulative:
            return w
    return next(iter(d))

def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original Text:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens)-2):
        if random.random() < 0.4: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spin Text:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))

if __name__ == '__main__':
    test_spinner()
