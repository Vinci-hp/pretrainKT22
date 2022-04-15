import numpy as np
import random
from random import random, shuffle

question_data_list = []
for i in range(10000):
    question_data = []
    for i in range(1, 102):
        question_data.append(i)
    shuffle(question_data)
    question_data_list.append(question_data)
length1 = len(question_data_list)
shuffle(question_data_list)
train_data = question_data_list
train_f = open('./assist2009_train_skill.csv', 'a', encoding='utf-8')
for train_d in train_data:
    pro = train_d
    p_ = ','.join('%s' %id for id in pro)
    train_f.write(p_ + '\n')

