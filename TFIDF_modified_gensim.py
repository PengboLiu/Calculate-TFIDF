from gensim import corpora
from gensim import models
# 以kaggle QIQC数据为例
X_train = ['how did quebec nationalists see their province as a nation in the ####s ? ',
'do you have an adopted dog ,  how would you encourage people to adopt and not shop ? ',
'why does velocity affect time ?  does velocity affect space geometry ? ']
print(X_train)

X_train_word_list = []
for i in range(len(X_train)):
    X_train_word_list.append(X_train[i].split(' '))

# 赋给语料库中每个词(不重复的词)一个整数id
# word_list = X_train_word_list+X_test_word_list
word_list = X_train_word_list
dictionary = corpora.Dictionary(word_list)
corpus = [dictionary.doc2bow(text) for text in word_list]

# 元组中第一个元素是词语在词典对应的id，第二个元素是词语在句子中出现的次数

tfidf = models.TfidfModel(corpus)
X_train_id = []  # 将句子表示成单词在词典中id的形式

word_id_dict = dictionary.token2id
for i in range(len(X_train_word_list)):
    sen_id = []
    word_sen = X_train_word_list[i]
    for j in range(len(word_sen)):
        id = word_id_dict.get(word_sen[j])
        if id is None:
            id = 0
        sen_id.append(id)
    X_train_id.append(sen_id)

# 现在的句子是用单词id表示成的，并且顺序是原始句子单词出现的顺序
print(X_train_id)

# 将id和tfidf值对应的形式存储到python的dict里，每一个句子都形成一个dict
X_train_tfidf_vec = []  # 每个句子是一个字典，key是单词的ID，value是单词对应的tfidf值

for i in range(len(X_train)):
    temp = {}
    string = X_train[i]
    string_bow = dictionary.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    # 每个句子是一个list，句中的每个单词表示为一个元组，元组的第一个元素是单词的ID，第二个元素是tfidf值

    for j in range(len(string_tfidf)):
#         print(string_tfidf[j][0])
        temp[string_tfidf[j][0]] = string_tfidf[j][1]
#         print(temp)
    X_train_tfidf_vec.append(temp)

# 最后我们将id表示的句子映射到tfidf值对应的句子，也就是把id去除

X_train_tfidf = []  # tfidf值形成的句子。每个句子是一个list
for i in range(len(X_train_id)):
    sen_id = X_train_id[i]
    sen_id_tfidf = X_train_tfidf_vec[i]
    sen = []
    for j in range(len(sen_id)):
        word_id = sen_id[j]
        word_tfidf = sen_id_tfidf.get(word_id)
        if word_tfidf is None:
            word_tfidf = 0
        sen.append(word_tfidf)
    X_train_tfidf.append(sen)

# 如果需要，可以做padding(keras)
# maxlen = 15
# x_train_tfidf = sequence.pad_sequences(X_train_tfidf, maxlen=maxlen,dtype='float64')