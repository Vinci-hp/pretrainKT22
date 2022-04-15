from torch.utils.data import Dataset
import numpy as np
import torch
from random import random, randint
from pretrain_model import Constants


def readTrainData(path, sq_length, sq_space):
    with open(path) as f:
        data = f.readlines()
    f.close()
    questions = []
    skills = []
    m = int(len(data)/4)
    for i in range(m):
        num = int(data[4*i].replace('\n', ''))
        list_q = data[4 * i + 1].replace('\n', '').split(',')
        list_q = list(map(lambda x: float(x), list_q))
        list_s = data[4 * i + 2].replace('\n', '').split(',')
        list_s = list(map(lambda x: float(x), list_s))
        if num >= sq_length:
            if num == sq_length:
                questions.append(list_q)
                skills.append(list_s)
            else:
                mod = num % sq_length
                m = int(num / sq_length) - 1
                windows = mod + sq_length * m
                for i in range(windows+1):
                    index = sq_space * i
                    if index > windows:
                        break
                    x_1 = list_q[index:index + (sq_length-1)]
                    s_1 = list_s[index:index + (sq_length-1)]
                    questions.append(x_1)
                    skills.append(s_1)

    return questions, skills


def readSkillData(path):
    with open(path) as f:
        data = f.readlines()
    f.close()
    skill_Data = []
    for i in range(len(data)):
        list_s = data[i].replace('\n', '').split(',')
        list_s = list(map(lambda x: float(x), list_s))
        skill_Data.append(list_s)
    return skill_Data


def get_difficulty(path):
    with open(path) as f:
        data1 = f.readlines()
    f.close()
    list_q = data1[0].replace('\n', '').split(',')
    list_diff = data1[1].replace('\n', '').split(',')
    assert len(list_q) == len(list_diff)
    dic = {}
    for i in range(len(list_q)):
        dic[list_q[i]] = list_diff[i]
    return dic


def getData(path, sq_len, sq_space):
    train_path = '../datasets/'+path+'/assist'+path+'_pid_train.csv'
    skill_path = '../datasets/'+path+'/assist'+path+'_train_skill.csv'
    difficulty_path = '../datasets/'+path+'/diffict_'+path+'.txt'

    training_data = readTrainData(train_path, sq_len, sq_space)
    training_skill_data = readSkillData(skill_path)
    difficulty_dict = get_difficulty(difficulty_path)

    return training_data, training_skill_data, difficulty_dict


def MaskedtrainData(data, ques_size):
    mask_questions = []
    question_labels = []
    quest, skill = data
    data_zip = zip(quest, skill)
    for data_ in data_zip:
        mask_q, q_label = Masked_single_question(data_, ques_size)

        mask_questions.append(mask_q)
        question_labels.append(q_label)

    return mask_questions, question_labels


def Masked_single_question(data, q_size):
    mask_q = []
    q_label = []
    threshold = 0.15
    quest, skill = data
    assert len(quest) == len(skill)
    num = len(quest)
    for i in range(num):
        r = random()
        if r < threshold:  # mask 15% of all token
            if r < threshold * 0.8:  # 80% replace [mask]=question_size+3
                mask_q.append(Constants.mask_Q+q_size)

            elif r < threshold * 0.9:  # 10% random (1, question_size)
                random_q_index = randint(1, q_size)
                mask_q.append(random_q_index)
            else:
                mask_q.append(quest[i])
            q_label.append(quest[i])
        else:
            mask_q.append(quest[i])
            q_label.append(Constants.PAD)

    return mask_q, q_label


def getTrainDiffData(data, total_skill, difficult_dict):
    questions = []
    total_skills = []
    difficulty_labels = []
    quest, skill = data
    data_zip = zip(quest, skill, total_skill)
    for data_ in data_zip:
        d_ques, total_skill, dif_label = get_single_dif(data_, difficult_dict)
        questions.append(d_ques)
        total_skills.append(total_skill)
        difficulty_labels.append(dif_label)
    return questions, total_skills, difficulty_labels


def get_single_dif(data, difficult_dict):
    question = []
    dif_label = []
    quest, skill, skill_total = data
    assert len(quest) == len(skill)
    num = len(quest)
    for i in range(num):
        question.append(quest[i])
        dif_label.append(float(difficult_dict[str(quest[i])]))
    return question, skill_total, dif_label


def Masked_single_skill(data, skill_size):
    mask_total_skill = []
    total_skill_label = []
    threshold = 0.15
    num = len(data)
    for i in range(num):
        r = random()
        if r < threshold:  # mask 15% of all token
            if r < threshold * 0.8:  # 80% replace [mask]=skill_size+3
                mask_total_skill.append(Constants.mask_C+skill_size)

            elif r < threshold * 0.9:  # 10% random (1, skill_size)
                random_s_index = randint(1, skill_size)
                mask_total_skill.append(random_s_index)
            else:
                mask_total_skill.append(data[i])
            total_skill_label.append(data[i])
        else:
            mask_total_skill.append(data[i])
            total_skill_label.append(Constants.PAD)

    return mask_total_skill, total_skill_label


def Masked_skill_trainData(data, skill_size):
    mask_total_skills = []
    total_skill_labels = []
    for data_ in data:
        mask_s, s_label = Masked_single_skill(data_, skill_size)
        mask_total_skills.append(mask_s)
        total_skill_labels.append(s_label)

    b_mask_total_skills = np.array(mask_total_skills, dtype=float)
    b_total_skill_labels = np.array(total_skill_labels, dtype=float)

    batch_mask_total_skills = torch.from_numpy(b_mask_total_skills).long()
    batch_total_skill_labels = torch.from_numpy(b_total_skill_labels).long()

    return batch_mask_total_skills, batch_total_skill_labels


def get_trainTensor(traindata, question_size, max_len):

    # =======mask===========
    mask_questions, question_label = MaskedtrainData(traindata, question_size)

    b_mask_questions = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in mask_questions], dtype=float)
    b_question_label = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in question_label], dtype=float)

    batch_mask_questions = torch.from_numpy(b_mask_questions).long()
    batch_question_label = torch.from_numpy(b_question_label).long()

    return batch_mask_questions, batch_question_label


def getDiffTrainData(traindata, dif_skilldata, difficult, max_len):

    d_questions, d_total_skills, dif_labels = getTrainDiffData(traindata, dif_skilldata, difficult)
    b_questions = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in d_questions], dtype=float)
    b_total_skills = np.array(d_total_skills, dtype=float)
    b_diff_label = np.array([e + [Constants.PAD_C] * (max_len - len(e)) for e in dif_labels], dtype=float)

    batch_d_questions = torch.from_numpy(b_questions).long()
    batch_d_total_skills = torch.from_numpy(b_total_skills).long()
    batch_diff_label = torch.from_numpy(b_diff_label).float()

    return batch_d_questions, batch_d_total_skills, batch_diff_label


class PretrainDataSet(Dataset):
    def __init__(self, data, total_skill_data, difficult_dict, question_size, skill_size, max_len):
        self.batch_mask_questions, self.batch_question_label = get_trainTensor(data, question_size, max_len=max_len)

        self.batch_mask_total_skill, self.batch_total_skill_label = Masked_skill_trainData(total_skill_data, skill_size)

        self.batch_question, self.batch_total_skill, self.diff_label = getDiffTrainData(data, total_skill_data, difficult_dict, max_len=max_len)

        assert (self.batch_mask_questions.size(0) == self.batch_mask_total_skill.size(0)
                == self.batch_question.size(0) == self.batch_total_skill.size(0))
        self.length = self.batch_mask_questions.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_mask_q = self.batch_mask_questions[idx]
        batch_q_label = self.batch_question_label[idx]
        batch_mask_s = self.batch_mask_total_skill[idx]
        batch_s_label = self.batch_total_skill_label[idx]
        batch_question = self.batch_question[idx]
        batch_total_skill = self.batch_total_skill[idx]
        batch_diff_label = self.diff_label[idx]

        return batch_mask_q, batch_q_label, batch_mask_s, batch_s_label, batch_question, batch_total_skill, batch_diff_label

#
# train_data, skill_data, dif = getData('2009', 20, 5)
# training_data = torch.utils.data.DataLoader(PretrainDataSet(train_data, skill_data, dif, 16891, 101, 20), batch_size=128)
# for b in training_data:
#     batch_mask_q, batch_q_label, batch_mask_s, batch_s_label, batch_question, batch_total_skill, batch_diff_label = b
#     print(batch_question[:2])
#     print(batch_mask_s[:2])
#     print(batch_total_skill.shape)
#     print(batch_total_skill[:2])
#     # print(batch_s_label[:3])
#     break

