#%%库引入
import os
import jieba
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd



#%%情感词典构建

stopwords=[line.strip() for line in open('C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/stopwords/stopword.txt', 'r',encoding='utf-8',errors='ignore').readlines()]
#结合了多个停用词表

def open_dict(Dict='hahah',path='C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/emotion_dict/'):
    path = path + '%s.txt' %Dict
    dictionary = open(path, 'r', encoding='utf-8-sig',errors='ignore')#encoding='utf-8-sig',检查是否有文件头，并去掉
    dict = []
    for word in dictionary:
        word=word.strip('\n')
        word=word.strip(' ')
        dict.append(word)
    return dict
posdict = open_dict(Dict='posdict')#积极情感词典
negdict = open_dict(Dict='negdict')#消极情感词典

f=open('C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/emotion_dict/酒店情感词典.txt','r',encoding='utf-8')
words = []
value=[]
for word in f.readlines():
    words.append(word.split(' ')[0])
    value.append(float(word.split(' ')[1].strip('\n')))
    
c={'words':words,
   'value':value}
fd=pd.DataFrame(c)
pos=fd['words'][fd.value>0]
posdict=posdict+list(pos)    ##加入酒店相关的正向情感词
neg=fd['words'][fd.value<0]
negdict=negdict+list(neg)    ##加入酒店相关的负向情感词
alldict=posdict+negdict
f.close()


#%%预处理函数

def remove_characters(sentence):  #去停用词
    cleanwordlist=[word for word in sentence if word.lower() not in stopwords]
    filtered_text=' '.join(cleanwordlist)
    return filtered_text

def remove_emotion_characters(sentence):  #抽取情感词
    wordlist=[word for word in sentence if word in alldict]
    filtered_text=' '.join(wordlist)
    return filtered_text

def text_normalize(text):
    text_split=[]
    for line in text:
        text_split.append(list(jieba.cut(line)))
    text_normal=[]
    for word_list in text_split:
        text_normal.append(remove_characters(word_list))
    return text_normal    


def text_normalize2(text):    #基于情感词典的预处理
    text_split=[]
    for line in text:
        text_split.append(list(jieba.cut(line)))
    text_normal=[]
    for word_list in text_split:
        text_normal.append(remove_emotion_characters(word_list))
    return text_normal    



def get_content(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        content=''
        for l in f:
            l=l.strip().replace(u'\u3000',u'')
            content+=l
    return content        


def get_file_content(path):
    flist=os.listdir(path)
    flist=[os.path.join(path,x) for x in flist]
    corpus=[get_content(x) for x in flist]
    return corpus

#读取语料文本
pos_comment=get_file_content('C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/6000/pos')
neg_comment=get_file_content('C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/6000/neg')



#%%词频统计、词云图

all_comment=''
for x in pos_comment:
    all_comment+=x
for x in neg_comment:
    all_comment+=x



split_words=list(jieba.cut(all_comment))  #分词

filtered_corpus=remove_characters(split_words)  #去停用词
filtered_corpus=[word for word in split_words if word not in stopwords]

##词频统计
word_counts = collections.Counter(filtered_corpus) # 对分词做词频统计
word_counts_top10 = word_counts.most_common(10) # 获取前10最高频的词
print (word_counts_top10) # 输出检查

##词云制作
wordcloud_1 = WordCloud(
        font_path='C:/Windows/Fonts/simkai.ttf',#设置字体，为电脑自带黑体
        max_words=200, # 最多显示词数
        max_font_size=100, # 字体最大值)
        background_color='white',
        mask=plt.imread('C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/house.jpg')
        )
wordcloud=wordcloud_1.generate_from_frequencies(word_counts)
wordcloud.to_file('C:/Users/dell/Desktop/文本挖掘期末/hotel_comment/wordcloud.jpg')
plt.figure(figsize=(16,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


#%%数据整理与划分

pos_lable=[1 for i in range(3000)]  #加入标签
neg_lable=[-1 for i in range(3000)]
comments=pos_comment+neg_comment   #语料整合
lables=pos_lable+neg_lable
c={'comment':comments,
   'value':lables}
df=pd.DataFrame(c)

x=df['comment']
y=df['value']

from sklearn.model_selection import train_test_split

#分割数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)
X_train=text_normalize(X_train)
X_test=text_normalize(X_test)

#X_train=text_normalize2(X_train)    #方法二：基于情感词典的特征提取
#X_test=text_normalize2(X_test)


#%%机器学习部分

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#TF-IDF计算权重
tfidf_vectorizer=TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=(1,1))
tf_train_features= tfidf_vectorizer.fit_transform(X_train)
tf_test_features= tfidf_vectorizer.transform(X_test)

#a=tfidf_vectorizer.vocabulary_
#b=sorted(a.items(),key=lambda item:item[1],reverse=True)



##预测函数
def train_predict_evaluate_model(classifier,train_features,train_labels,test_features,test_labels):
    classifier.fit(train_features,train_labels)
    predictions=classifier.predict(test_features)
    return predictions


##贝叶斯
mnb=MultinomialNB()
mnb_tf_pre=train_predict_evaluate_model(classifier=mnb,train_features=tf_train_features,train_labels=y_train,test_features=tf_test_features,test_labels=y_test)


##SVM
svm=SGDClassifier(loss='hinge',max_iter=1000)
svm_tf_pre=train_predict_evaluate_model(classifier=svm,train_features=tf_train_features,train_labels=y_train,test_features=tf_test_features,test_labels=y_test)


###ann

ann_model = MLPClassifier(hidden_layer_sizes=1, activation='logistic', solver='lbfgs', random_state=0)
ann_tf_pre=train_predict_evaluate_model(classifier=ann_model,train_features=tf_train_features,train_labels=y_train,test_features=tf_test_features,test_labels=y_test)


##逻辑回归
logreg = LogisticRegression(C=1,penalty='l2')
log_tf_pre=train_predict_evaluate_model(classifier=logreg,train_features=tf_train_features,train_labels=y_train,test_features=tf_test_features,test_labels=y_test)


#准确率
print('贝叶斯准确率：',accuracy_score(y_test, mnb_tf_pre))

print('SVM准确率：',accuracy_score(y_test, svm_tf_pre))

print('ann准确率：',accuracy_score(y_test, ann_tf_pre))

print('逻辑回归准确率：',accuracy_score(y_test, log_tf_pre))



#%%基于情感词典的分析
from emotion_score import sentiment
predictions=[]
for line in comments:
    predictions.append(sentiment(line))
from sklearn.metrics import accuracy_score
print('基于情感词典的准确率：',accuracy_score(lables, predictions))


  



