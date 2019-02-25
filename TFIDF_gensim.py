from gensim import corpora
from gensim import models

corpus = [
        'this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document'
    ]

word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
print(word_list)

# 赋给语料库中每个词(不重复的词)一个整数id
dictionary = corpora.Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
# 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
print(new_corpus)
# 通过下面的方法可以看到语料库中每个词对应的id
print(dictionary.token2id)

tfidf = models.TfidfModel(new_corpus)
tfidf.save("my_model.tfidf")
# 载入模型
tfidf = models.TfidfModel.load("my_model.tfidf")
# 使用这个训练好的模型得到单词的tfidf值
tfidf_vec = []
for i in range(len(corpus)):
    string = corpus[i]
    string_bow = dictionary.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)
print(tfidf_vec)

'''
gensim训练出来的tf-idf值左边是词的id，右边是词的tfidf值
gensim有自动去除停用词的功能，比如the
gensim会自动去除单个字母，比如i
gensim会去除没有被训练到的词，比如name
所以通过gensim并不能计算每个单词的tfidf值
'''