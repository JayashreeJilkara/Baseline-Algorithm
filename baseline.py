import sys
import pandas as pd
import nltk
import re
from nltk import bigrams

def dataSet(Path):
    with open(Path,'r') as text:
        List_Of_Words = []
        List_Of_Tags = []
        Overall_List = []
        for x in text.readlines():
            for y in x.split():
                term, sym, pos_Tag = y.partition('/')
                Overall_List.append((term,pos_Tag))
                List_Of_Words.append(term)
                List_Of_Tags.append(pos_Tag)
    return Overall_List,List_Of_Words,List_Of_Tags

train_set,train_word_List, train_tag_List = dataSet('POS.train')
test_set, test_word_List, test_tag_List = dataSet('POS.test')


train_setTokenCount ={}
for i in train_set:
    if(i not in train_setTokenCount):
        train_setTokenCount[i]=1
    else:
        train_setTokenCount[i]+=1

#print(train_setTokenCount)

def maxTag(word):
    try:
        l=[]
        l1=[]
        for k,v in train_setTokenCount.items():
            if(k[0]==word):
                l.append(k[1])
                l1.append(v)
        c= l[l1.index(max(l1))]
        return (word,c)
        #print((word,c))
    except:
        return (word,'NN')

#maxTag('owned')

predList = []
for i in test_word_List:
    ans = maxTag(i)
    predList.append(ans)
    #print(predList)

print(predList)

#print(test_set)

r = 0
w = 0
for i in range(len(test_set)):
  a = test_set[i]
  b = predList[i]
  for z in range(len(a)):
    if a[z] == b[z]:
      r = r+1
    else:
      w = w +1

#print(r)
#print(w)
print('Accuracy on trainset is: ',(r/(r+w))*100,'%')
#print('Loss is: ',(w/(r+w))*100,'%')

