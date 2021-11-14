from utils import *
##### LOAD DATA INFO #####
import pandas as pd

df = pd.read_csv("/Users/david/Documents/TED Talk + Glassdoor NLP Data and Code/ted_data/.cleaned ted talks w LDA(04.28.21).csv.icloud")
list(df)
import matplotlib.pyplot as plt

plt.hist(df["wordcount"])  # highest wordcount 9360

df["ratings"]

# PRONOUNS -- TRACK AND REMOVE


count_list = list()
for text in df["transcript"]:
    i_count = pronoun_counter1(text)
    count_list.append(i_count)
df["i_count"] = count_list

count_list = list()
for text in df["transcript"]:
    not_i_count = pronoun_counter2(text)
    count_list.append(not_i_count)
df["not_i_count"] = count_list


df["transcript_np"] = df.transcript.apply(rem_pronoun)

# LOAD DATA
# import pandas as pd
# df1 = pd.read_csv("C:/Users/cbh2132/Documents/transcripts.csv")
# list(df1)
# df2 = pd.read_csv("C:/Users/cbh2132/Documents/ted_main.csv")
# list(df2)

# JOIN TRANSCRIPTS AND TRACKING VARIABLES
# df = df.join(df2.set_index('url'), on='url')
# list(df)

# CLEANING TRANSCRIPTS
''' CLEANING DECISIONS:
    1. We decontracts (e.g. "hasn't" becomes "has not")
    2. We parentheticals (i.e. sound effects, music, and applause)
    3. We removed special chars (i.e. anything not "A-z")
    4. Non-English words removed
    5. We stemmed the texts
    6. Removed most stopwords but kept pronouns
    7. FOR LDA, we removed any text that was empty or under 50 words (there were 10 of these)'''
# df["transcript"] = df.transcript.apply(decontracted)
# df["og_wordcount"] = df.transcript.apply(wordcount) # original wordcount
# df["transcript"] = df.transcript.apply(rem_sw)
# df["transcript"] = df.transcript.apply(clean_text)
# df["wordcount"] = df.transcript.apply(wordcount) # wordcount after cleaning
# df["transcript"] = df.transcript.apply(rem_sw)
# df = df[df['wordcount'] >= 50]
# spotcheck
# df.transcript[3]
# WRITE CLEANED DF TO NEW CSV
# df.to_csv("cleaned ted talks.csv", encoding='utf-8', index=False)


# CREATE RATIOS OF SPECIFIC RATINGS TO OVERALL RATINGS
import ast

ratios = []
for i in range(0, df.shape[0]):
    test = ast.literal_eval(df.iloc[i]['ratings'])
    rating_count = 0
    inspiring_count = 0
    for rating in test:
        rating_count += rating['count']
        if (rating['name'] == 'Inspiring'):
            inspiring_count = rating['count']
    ratios.append(inspiring_count / rating_count)
df["inspiring"] = ratios

ratios = []
for i in range(0, df.shape[0]):
    test = ast.literal_eval(df.iloc[i]['ratings'])
    rating_count = 0
    inspiring_count = 0
    for rating in test:
        rating_count += rating['count']
        if (rating['name'] == 'Persuasive'):
            inspiring_count = rating['count']
    ratios.append(inspiring_count / rating_count)
df["persuasive"] = ratios

ratios = []
for i in range(0, df.shape[0]):
    test = ast.literal_eval(df.iloc[i]['ratings'])
    rating_count = 0
    inspiring_count = 0
    for rating in test:
        rating_count += rating['count']
        if (rating['name'] == 'Unconvincing'):
            inspiring_count = rating['count']
    ratios.append(inspiring_count / rating_count)
df["Unconvincing"] = ratios

#### CREATE LDA FEATURES ####

# LDA
# REFERENCE CODE: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
import gensim
from gensim import models

# data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(df["transcript_np"]))
import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_words]
from pprint import pprint

# Number of Topics
num_topics = 25
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       passes=100)
# Print the Keyword in the topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
# save model to disk (no need to use pickle module)
lda_model.save('lda.model25-100pass')
# later on, load trained model from file
model_s = models.LdaModel.load('lda.model25-100pass')
model_b = models.LdaModel.load('lda.model50-100pass')
model_s.print_topic(0, topn=20)
model_b.print_topic(0, topn=20)

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# TOPIC 50/PASS 100 = .-8.90 perplexity

# Compute Coherence Score
from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)  # we basically want above .5
# TOPIC 20/PASS 10 = .27 coherence
# TOPIC 20/PASS 50 = .31 coherence
# TOPIC 25/PASS 100 = .36 coherence
# TOPIC 30/PASS 50 = .32 coherence
# TOPIC 40/PASS 10 = .26 coherence
# TOPIC 50/PASS 50 = .32 coherence
# TOPIC 50/PASS 100 = .33 coherence
# TOPIC 60/PASS 50 = .32 coherence
# TOPIC 70/PASS 50 = .33 coherence


# VISUALIZE LDA OUTPUT AS HTML-PAGE
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
# Because I can't get vis to show up in the spyder IDE, export it as an html webpage with this code
pyLDAvis.save_html(vis, 'lda25100.html')

''' Plotting used to DETERMINED AN APPROPRIATE NUMBER OF TOPICS
# This is from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    from gensim.models.ldamodel import LdaModel
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes = 4)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words, start=10, limit=70, step=2)
# Show graph
import matplotlib.pyplot as plt
limit=70; start=10; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

'''

### BASIC SENTIMENT WORD-SCORES ####

# Ran this code on the uncleaned_transcripts to compose four sentiment features
'''
# SENTIMENT ANALYZER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
sentiment = SentimentIntensityAnalyzer() 
# this analyzer uses grammar, emoticons, etc
test = sentiment.polarity_scores("That is hot.")

#positive sentiment : (compound score >= 0.05)
#neutral sentiment : (compound score > -0.05) and (compound score < 0.05)
#negative sentiment : (compound score <= -0.05)
print(test); print(test["compound"])

pos = list()
neg = list()
neut = list()
comp = list()
counter = 0

for x in df["unclean_transcript"]:
    counter += 1
    print(counter)
    test = sentiment.polarity_scores(str(x))
    neg.append(test["neg"])
    neut.append(test["neu"])
    pos.append(test["pos"])
    comp.append(test["compound"])

list(df)
df["pos"] = pos
df["neg"] = neg
df["neut"] = neut
df["comp"] = comp
'''

# ADD LDA TOPICS TO DATAFRAME
data = df["transcript_np"]


# Run function to create new df
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)
# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
# Check words in topic
lda_model.print_topic(24, topn=20)

# This command prints out the topics in a text and the percent of the text from each topic
lda_model[corpus[0]]
lda_model[corpus[1]]
# if you want to make a list for each doc-topic pair
# l = [lda_model.get_document_topics(item) for item in corpus]

# Create a list of lists (list of topics inside each doc)
topic_distribution = list()
for num in range(len(df["transcript_np"])):
    props = [0] * 25  # create empty list
    doc = num
    for item in lda_model[corpus[num]]:
        topic = item[0]
        props[topic] = item[1]
    topic_distribution.append(props)
# Turn list of lists into dataframe
df2 = pd.DataFrame(topic_distribution)
df = df.join(df2)

# WRITE CLEANED DF TO NEW CSV
# df.to_csv("cleaned ted talks w LDA.csv", encoding='utf-8', index=False)


# ANALYSIS AND ML ALGORITHMS -- LOAD DATA AND LDA MODEL
import pandas as pd
from gensim import models

lda_model = models.LdaModel.load('lda.model25-100pass')
df = pd.read_csv("/Users/david/Documents/TED Talk + Glassdoor NLP Data and Code/ted_data/.cleaned ted talks w LDA(04.28.21).csv.icloud")
list(df)

# CORRELATIONS
import scipy.stats

scipy.stats.pearsonr(df["inspiring"], df["views"])
scipy.stats.pearsonr(df["persuasive"], df["views"])
scipy.stats.pearsonr(df["longwinded"], df["views"])
scipy.stats.pearsonr(df["Unconvincing"], df["views"])
scipy.stats.pearsonr(df["jaccard_sim"], df["views"])
scipy.stats.pearsonr(df["jaccard_sim"], df["persuasive"])
list(df)

# How many talks are "inspiring" (i.e. 20% of the ratings are inspiring)
counter = 0
for x in df["persuasive"]:
    if x > .15:
        counter += 1
counter / len(df["persuasive"])

counter = 0
for x in df["Unconvincing"]:
    if x > .15:
        counter += 1
counter / len(df["Unconvincing"])

counter = 0
for x in df["longwinded"]:
    if x > .15:
        counter += 1
counter / len(df["longwinded"])

df["ratings"]

# CREATE BINARY OUTCOME FOR PERSUASIVE
counter = list()
for x in df["persuasive"]:
    if x > .15:
        counter.append(1)
    else:
        counter.append(0)
df["persuasive_b"] = counter
# SUBTRACT UNCONVINCING
counter = list()
for x in df["Unconvincing"]:
    if x > .15:
        counter.append(1)
    else:
        counter.append(0)
df["persuasive_b"] = df["persuasive_b"] - counter
# ADD INSPIRING
counter = list()
for x in df["inspiring"]:
    if x > .15:
        counter.append(1)
    else:
        counter.append(0)
df["persuasive_b"] = df["persuasive_b"] + counter
# SUBTRACT Longwinded
counter = list()
for x in df["longwinded"]:
    if x > .15:
        counter.append(1)
    else:
        counter.append(0)
df["persuasive_b"] = df["persuasive_b"] - counter
# THEN RE-BINARY
counter = list()
for x in df["persuasive_b"]:
    if x > 0:
        counter.append(1)
    else:
        counter.append(0)
df["persuasive_b"] = counter

# GENERAL DISTRIBUTION OF Y VARIABLE
df["persuasive_b"].describe()
df["persuasive_b"].value_counts()

# DATAFRAME ONLY WITH RELEVANT FEATURES
list(df)
df.columns.get_loc("0")  # index of first lda feature?
df.columns.get_loc("24")  # index of last lda feature?
features = df.iloc[0:2457, 32:56]
features["i_count"] = df["i_count"]
features["og_wordcount"] = df["og_wordcount"]
features["jaccard_sim"] = df["jaccard_sim"]
features["pos"] = df["pos"]
features["neut"] = df["neut"]
features["neg"] = df["neg"]
list(features)

# Training a Model w/ Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Comose training data, training labels, test data, and test labels
x_train, x_test, y_train, y_test = train_test_split(
    features, df.persuasive_b, test_size=.2, random_state=321)

# Run Random Forest
clf = RandomForestClassifier(max_depth=9, random_state=321)
# Build model by using "clf.fit(x, y)" where x is dataframe, y is what you want to predict
clf.fit(x_train, y_train)

# test classifier on new data
y_pred = clf.predict(x_test)
# CHECK PERFORMANCE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# CONFUSION MATRIX
confusion_matrix(y_test, y_pred)
# Return precision score, recall score, and f_score
precision_recall_fscore_support(y_test, y_pred, average="weighted")

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, x_test, y_test)
plt.show()

# You can get a list of feature importance (higher numbers = more important)
importance = list()
for x in clf.feature_importances_:
    importance.append(x)
importance
# Create new pandas DataFrame
f_imp = pd.DataFrame()
f_imp["names"] = list(features)
f_imp["importance"] = importance
f_imp.sort_values(by=['importance'])
# WORDS IN TOPIC
lda_model.print_topic(13, topn=50)

# DATAFRAME ONLY WITH RELEVANT FEATURES
list(df)
df.columns.get_loc("13")  # index of first lda feature?
features = pd.DataFrame({'13': df["13"]})
features["i_count"] = df["i_count"]
features["1"] = df["1"]
features["21"] = df["21"]
features["17"] = df["17"]
features["11"] = df["11"]
features["og_wordcount"] = df["og_wordcount"]
list(features)

# Training a Model w/ Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Comose training data, training labels, test data, and test labels
x_train, x_test, y_train, y_test = train_test_split(
    features, df.persuasive_b, test_size=.2, random_state=321)

# Run Random Forest
clf = RandomForestClassifier(max_depth=9, random_state=321)
# Build model by using "clf.fit(x, y)" where x is dataframe, y is what you want to predict
clf.fit(x_train, y_train)

# test classifier on new data
y_pred = clf.predict(x_test)
# CHECK PERFORMANCE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# CONFUSION MATRIX
confusion_matrix(y_test, y_pred)
# Return precision score, recall score, and f_score
precision_recall_fscore_support(y_test, y_pred, average="weighted")

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, x_test, y_test)
plt.show()

# You can get a list of feature importance (higher numbers = more important)
importance = list()
for x in clf.feature_importances_:
    importance.append(x)
importance
# Create new pandas DataFrame
f_imp = pd.DataFrame()
f_imp["names"] = list(features)
f_imp["importance"] = importance
f_imp.sort_values(by=['importance'])
# WORDS IN TOPIC
lda_model.print_topic(13, topn=50)
lda_model.print_topic(21, topn=50)
lda_model.print_topic(1, topn=50)

# PLOT AUC CURVE
from sklearn.metrics import plot_roc_curve

clf.fit(x_train, y_train)
clf_disp = plot_roc_curve(clf, x_test, y_test)
plt.show()
