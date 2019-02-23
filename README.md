# Calculate-TFIDF
计算TFIDF的三种方法：Python、sklearn、gensim

### Python原生
直接根据公式手写代码，灵活度高，但是速度较慢。

### 调用sklearn
sklearn提供了TfidfVectorizer这个接口来计算tfidf值。这个接口会返回一个存储tfidf值的稀疏矩阵。将这个稀疏矩阵转化成稠密矩阵或者array的话，就会占用内存过大，不可取。  
不建议使用sklearn的方法。

### 调用gensim
将每个句子对应到一个列表，每个列表里是句长个数的元组，每个元组是单词id和tfidf值的组合。id是通过gensim里面的一个接口获得的。  
有一点需要处理的是，每个句子对应的列表里元组出现的顺序并不是原始句子中单词出现的顺序，而是按照id的大小排序的。  
如果我们想要将得到的tfidf值出现的顺序变为单词在句中出现的顺序，需要将得到gensim形式的列表处理一下。
