#New Classifier
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import spacy
from nltk.stem.lancaster import LancasterStemmer
import json


parser = spacy.load('en_core_web_sm')
punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
st = LancasterStemmer()


class pipelineCleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        return [sanitize(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def computeAccuracy(test_data, output_data):     
    if len(test_data) != len(output_data):
       raise ValueError('Test data and output data are of uneven size!')
    else:
       return float(sum([1 for i, item in enumerate(output_data) if item == test_data[i]]))/len(test_data)


def sanitize(text):     
    return text.strip().lower()


def englishTokenizer(sentence):
    tokens = parser(sentence)
    tokens = [st.stem(tok.lemma_.lower().strip()) if st.stem(tok.lemma_.lower().strip()) != "-pron-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens


def checkExistingFiles():
    filesExist = [False, False]
    try:
        with open('training_data.txt', 'r') as fp:
            filesExist[0] = True
    except FileNotFoundError:
        pass

    try:
        with open('current_training_data.txt', 'r') as fp:
            filesExist[1] = True
    except FileNotFoundError:
        pass
    return filesExist


# muagi ang data, sa pipe para ma clean, vectorize, ug classify
pipe = Pipeline([
    ("cleaner", pipelineCleaner()),
    ('vectorizer', CountVectorizer(tokenizer = englishTokenizer, ngram_range=(1,1))),
    ('to_dense', DenseTransformer()), 
    ('classifier', GaussianNB())])

# training and testing data
training_data = []
current_training_data = []

hasData, hasOtherData = checkExistingFiles()
#check if bot has any data to base on
if not hasData:
    print('I know nothing, teach me responses to at least 10 statements for a headstart')
    with open('training_data.txt', 'w') as fw:
        fw.write('[')
        for n in range(10):
            statement = input('{0} #{1}: '.format('Statement', n+1))
            answer = input('{0} #{1}: '.format('Answer to statement', n+1))
            fw.write('\n    ["{0}", "{1}"],'.format(statement, answer))
            training_data.append([statement, answer])
        fw.write('\n]')
    pipe.fit([x[0] for x in training_data], [x[1] for x in training_data])
else:
    with open('training_data.txt', 'r') as fp:
        training_data = eval(fp.read())
    if hasOtherData:
        # open one file, store fp.read() to a variable
        with open('training_data.txt', 'r') as fp:
          original = fp.read()
        # open the other file, store fp.read() to another variable
        with open('current_training_data.txt', 'r') as fp2:
          current = fp2.read()
        # var1 == var2 ?
        # yes -> do nothing
        # no -> copy items rewrite the first txt file to be a copy of the other
        #    -> train
        if original != current:
          with open('training_data.txt', 'w') as fp:
            fp.write(current)
        pipe.fit([x[0] for x in training_data], [x[1] for x in training_data])

    else:
        # copy original to the current_training_data file
        with open('training_data.txt', 'r') as fp:
          original = fp.read()
        with open('current_training_data.txt', 'w') as fp2:
          fp2.write(original)
        pipe.fit([x[0] for x in training_data], [x[1] for x in training_data])

botResponse = ['', 0]
print('Enter "Teach Me" to input new questions and what should the bot respond with.')
while botResponse[0] != 'See you later, thanks for visiting':
    userInput = input('> ')
    #checks if user input is trying to teach the bot or just asking for help
    if userInput != 'Teach Me':
        botResponse = [pipe.predict([userInput])[0], 0]
        with open('training_data.txt', 'r') as f:
            original = f.readlines()

        #dynamic error threshold
        # if len(original) <= 50:
        #     threshold = 0.10
        # elif len(original) <= 150:
        #     threshold = -0.10
        # elif len(original) <= 250:
        #     threshold = -0.30
        # elif len(original) <= 350:
        #     threshold = -0.50
        # elif len(original) <= 450:
        #     threshold = -0.70
        # elif len(original) >450:
        #     threshold = -0.90

        #checks if the bot has a definitive answer to the question
        if botResponse[1] < -0.90:
            print('I dont have a good enough response for that, please tell me what should I respond to that kind of question/query so that I can learn it:')
            userInput2 = input('> ')
            print('Alrighty All Set! If I get the question/query "{}"" or anything similar to it, I will reply with "{}"'.format(userInput, userInput2))
            with open('training_data.txt', 'r') as file:
                original = file.read()[:-1]
            with open('current_training_data.txt', 'w') as file2:
                file2.write(original)
                file2.write('    ["{0}", "{1}"],'.format(userInput, userInput2))
                file2.write('\n]')
            with open('current_training_data.txt', 'r') as file3:
                current = file3.read()
            with open('training_data.txt', 'w') as file4:
                file4.write(current)
                training_data = eval(current)
            pipe.fit([x[0] for x in training_data], [x[1] for x in training_data])

        else:
            print ('Bot: {0} (Confidence: {1})'.format(botResponse[0], botResponse[1]))


    else:
        print('How many questions and responses should I learn?')
        noOfNewQuestions = int(input('> '))
        with open('training_data.txt', 'r') as fp:
            original = fp.read()[:-1]
        with open('current_training_data.txt', 'w') as fw:
            fw.write(original)
            for n in range(noOfNewQuestions):
              statement = input('{0} #{1}: '.format('Statement', n+1))
              answer = input('{0} #{1}: '.format('Answer to statement', n+1))
              if n == 0:
                fw.write('    ["{0}", "{1}"],'.format(statement, answer))
                current_training_data.append([statement, answer])
              else:
                fw.write('\n    ["{0}", "{1}"],'.format(statement, answer))
                current_training_data.append([statement, answer])
            fw.write('\n]')
        with open('current_training_data.txt', 'r') as fw2:
            current = fw2.read()
        with open('training_data.txt', 'w') as fp2:
            fp2.write(current)
        training_data = eval(current)
        pipe.fit([x[0] for x in training_data], [x[1] for x in training_data])
