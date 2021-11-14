def decontracted(phrase):
    import re
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def ps_stem(var):
    # load stuff you need to clean
    from nltk.stem import PorterStemmer
    import enchant
    d = enchant.Dict("en_US")
    ps = PorterStemmer()
    # clean
    tmp = var.split()  # create an array from each row/ tokenize
    fin_var = [word for word in tmp if d.check(word)]  # remove non-english
    fin_var = [ps.stem(word) for word in fin_var]  # stem
    fin_var = " ".join(fin_var)  # rejoin tokens to sentences
    return fin_var


def clean_text(text):
    import re
    clean_text = re.sub("[\(\[].*?[\)\]]", "", text)
    clean_text = re.sub('[^A-z]+', " ", clean_text).lower()
    return clean_text


def rem_sw(var):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    pronouns = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves']
    sw = [x for x in sw if x not in pronouns]
    tmp = var.split()  # create an array from each row/ tokenize
    fin_var = [word for word in tmp if word not in sw]  # remove stopwords
    fin_var = " ".join(fin_var)  # rejoin tokens to sentences
    return fin_var


def wordcount(text):
    count = 0
    for word in text.split():
        count += 1
    return count


def tokenize(col):
    token_list = list()
    for x in col:
        x = x.split()
        token_list.append(x)
    return (token_list)


# How similar are two documents?
def jaccard_func(doc_a, doc_b):
    doc_a_set = set(doc_a.split())
    doc_b_set = set(doc_b.split())
    j_d = float(len(doc_a_set.intersection(doc_b_set))) / float(len(doc_a_set.union(doc_b_set)))
    return j_d


def jc_column(df_column_of_texts):
    jc_sim = list()
    counter = 0
    for x in df_column_of_texts:
        counter += 1
        print(counter)
        sim_score = -1
        for y in df_column_of_texts:
            sim_score += jaccard_func(str(x), str(y))
        ave = sim_score / (len(df_column_of_texts) - 1)
        print(ave)
        jc_sim.append(ave)
    return jc_sim

def pronoun_counter1(text):
    i_count = 0
    i_pronouns = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves']
    from nltk import word_tokenize
    x = word_tokenize(text)
    for word in x:
        if word in i_pronouns:
            i_count += 1
    return (i_count)


def pronoun_counter2(text):
    not_i_count = 0
    not_i_pronouns = ['you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                      'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'they', 'them', 'their',
                      'theirs', 'themselves']
    from nltk import word_tokenize
    x = word_tokenize(text)
    for word in x:
        if word in not_i_pronouns:
            not_i_count += 1
    return (not_i_count)

def sent_to_words(sentences):
    import gensim
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def rem_pronoun(text):
    pronouns = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves']
    text = [word for word in text.split() if word not in pronouns]
    text = " ".join(text)
    return text

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)
