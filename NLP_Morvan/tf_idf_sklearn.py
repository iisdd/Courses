# 使用scikit-learn内置的tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from visual import show_tfidf   # this refers to visual.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)


docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)
# 打印所有词汇的idf
print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
# 打印词汇对应下标
print("v2i: ", vectorizer.vocabulary_)


q = "I get a coffee cup"
qtf_idf = vectorizer.transform([q])

print(qtf_idf)          # 显示结果为:
                        #    (0, 12)	0.7550665514877668
                        #    (0, 11)	0.6556481547479347
                        # 即只有coffee,cup对应的地方有值,稀疏矩阵的存储方法
res = cosine_similarity(tf_idf, qtf_idf)
res = res.ravel().argsort()[-3:]
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in res[::-1]]))


i2v = {i: v for v, i in vectorizer.vocabulary_.items()}
dense_tfidf = tf_idf.todense()          # 从稀疏表达方式变回完整的矩阵
print(dense_tfidf.shape)                # (15, 44)
show_tfidf(dense_tfidf, [i2v[i] for i in range(dense_tfidf.shape[1])], "tfidf_sklearn_matrix")