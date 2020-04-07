import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_in_path', type=str, default='./raw_data/Baike/', help='the input path of data')
args = parser.parse_args()
data_in_path = args.data_in_path
in_path = "./raw_data/Baike/"
out_path = "./data"
case_sensitive = False
if not os.path.exists('./data'):
    os.mkdir('./data')
train_file_name = data_in_path + 'train.json'
test_file_name = data_in_path + 'test.json'
word_file_name = in_path + 'word_vec.json'
char_file_name = in_path + 'char_vec.json'
rel_file_name = in_path + 'rel2id.json'
pos_file_name = in_path + 'pos2id.json'
deprel_file_name = in_path + 'dep_vec.json'


def find_pos(sentence, head, tail):
    def find(sentence, entity):
        p = sentence.find(' ' + entity + ' ')
        if p == -1:
            if sentence[:len(entity) + 1] == entity + ' ':
                p = 0
            elif sentence[-len(entity) - 1:] == ' ' + entity:
                p = len(sentence) - len(entity)
            else:
                p = 0
        else:
            p += 1
        return p

    sentence = ' '.join(sentence.split())
    p1 = find(sentence, head)
    p2 = find(sentence, tail)
    words = sentence.split()
    cur_pos = 0
    pos1 = -1
    pos2 = -1
    for i, word in enumerate(words):
        if cur_pos == p1:
            pos1 = i
        if cur_pos == p2:
            pos2 = i
        cur_pos += len(word) + 1
    return pos1, pos2


def init(file_name, word_vec_file_name, char_vec_file_name, rel2id_file_name, pos2id_file_name, deprel_file_name,
         max_length=80, MDP_length=15, case_sensitive=False, is_training=True):
    print(file_name)
    if file_name is None or not os.path.isfile(file_name):
        raise Exception("[ERROR] Data file doesn't exist")
    if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
        raise Exception("[ERROR] Word vector file doesn't exist")
    if char_vec_file_name is None or not os.path.isfile(char_vec_file_name):
        raise Exception("[ERROR] Char vector file doesn't exist")
    if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
        raise Exception("[ERROR] rel2id file doesn't exist")
    if pos2id_file_name is None or not os.path.isfile(pos2id_file_name):
        raise Exception("[ERROR] pos2id file doesn't exist")
    if deprel_file_name is None or not os.path.isfile(deprel_file_name):
        raise Exception("[ERROR] dep rel file doesn't exist")
    char_num = 4
    path_num = 7
    print("Loading data file...")
    ori_data = json.load(open(file_name, "r"))
    print("Finish loading")
    print("Loading word_vec file...")
    ori_word_vec = json.load(open(word_vec_file_name, "r"))
    print("Loading char_vec file...")
    ori_char_vec = json.load(open(char_vec_file_name, "r"))
    print("Finish loading")
    print("Loading rel2id file...")
    rel2id = json.load(open(rel2id_file_name, "r"))
    print("Finish loading")
    print("Loading pos tag file...")
    pos2id = json.load(open(pos2id_file_name, "r"))
    pos2id['UNK'] = len(pos2id)
    pos2id['BLANK'] = len(pos2id)
    print("Finish loading")
    print('Loading dep rel_vec file...')
    dep_rel_vec = json.load(open(deprel_file_name, 'r'))
    print('Finish loading')

    if not case_sensitive:
        print("Eliminating case sensitive problem...")
        for i in ori_data:
            i['sentence'] = i['sentence'].lower()
            i['head']['word'] = i['head']['word'].lower()
            i['tail']['word'] = i['tail']['word'].lower()
        for i in ori_word_vec:
            i['word'] = i['word'].lower()
        print("Finish eliminating")

    # vec
    print("Building word vector matrix and mapping...")
    word2id = {}
    word_vec_mat = []
    word_size = len(ori_word_vec[0]['vec'])
    print("Got {} words of {} dims".format(len(ori_word_vec), word_size))
    for i in ori_word_vec:
        word2id[i['word']] = len(word2id)
        word_vec_mat.append(i['vec'])
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    word_vec_mat.append(np.random.normal(loc=0, scale=0.05, size=word_size))
    word_vec_mat.append(np.zeros(word_size, dtype=np.float32))
    word_vec_mat = np.array(word_vec_mat, dtype=np.float32)
    print("Finish building")

    print("Building char vector matrix and mapping...")
    char2id = {}
    char_vec_mat = []
    char_size = len(ori_char_vec[0]['vec'])
    print("Got {} chars of {} dims".format(len(ori_char_vec), char_size))
    for i in ori_char_vec:
        char2id[i['word']] = len(char2id)
        char_vec_mat.append(i['vec'])
    char2id['UNK'] = len(char2id)
    char2id['BLANK'] = len(char2id)
    char_vec_mat.append(np.random.normal(loc=0, scale=0.05, size=char_size))
    char_vec_mat.append(np.zeros(char_size, dtype=np.float32))
    char_vec_mat = np.array(char_vec_mat, dtype=np.float32)
    print("Finish building")

    print("Building dep rel vector matrix and mapping...")
    deprel2id = {}
    deprel_vec_mat = []
    deprel_size = len(dep_rel_vec[0]['vec'])
    print("Got {} words of {} dims".format(len(dep_rel_vec), deprel_size))
    for i in dep_rel_vec:
        deprel2id[i['word']] = len(deprel2id)
        deprel_vec_mat.append(i['vec'])
    deprel2id['UNK'] = len(deprel2id)
    deprel2id['BLANK'] = len(deprel2id)
    deprel_vec_mat.append(np.random.normal(loc=0, scale=0.05, size=deprel_size))
    deprel_vec_mat.append(np.zeros(deprel_size, dtype=np.float32))
    deprel_vec_mat = np.array(deprel_vec_mat, dtype=np.float32)
    print("Finish building")

    # sorting
    print("Sorting data...")
    ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
    print("Finish sorting")

    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_num), dtype=np.int64)
    char_mask = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_pos1 = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_pos2 = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_tag = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_mask = np.zeros((sen_tot, max_length, 3), dtype=np.float32)
    MDP_word = np.zeros((sen_tot, path_num, MDP_length), dtype=np.int64)
    MDP_rel = np.zeros((sen_tot, path_num, MDP_length), dtype=np.int64)
    MDP_pos = np.zeros((sen_tot, path_num, MDP_length), dtype=np.int64)
    MDP_dir = np.zeros((sen_tot, path_num, MDP_length), dtype=np.int64)
    sen_label = np.zeros((sen_tot), dtype=np.int64)
    head_ent = np.zeros((sen_tot), dtype=np.int64)
    tail_ent = np.zeros((sen_tot), dtype=np.int64)
    root_verb = np.zeros((sen_tot), dtype=np.int64)
    sen_len = np.zeros((sen_tot), dtype=np.int64)
    MDP_len = np.zeros((sen_tot, path_num), dtype=np.int64)
    bag_label = []
    bag_scope = []
    bag_key = []

    for i in range(len(ori_data)):
        if i % 1000 == 0:
            print(i)
        sen = ori_data[i]
        # sen_label
        if sen['relation'] in rel2id:
            sen_label[i] = rel2id[sen['relation']]
        else:
            sen_label[i] = rel2id['NA']

        words = sen['sentence'].split()
        # sen_len
        sen_len[i] = min(len(words), max_length)
        # sen_word
        for j, word in enumerate(words):
            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']
                for k, char in enumerate(word):
                    if k < char_num:
                        if char in char2id:
                            sen_char[i][j][k] = char2id[char]
                        else:
                            sen_char[i][j][k] = char2id['UNK']
                char_mask[i][j] = k + 1
                for k in range(k + 1, char_num):
                    sen_char[i][j][k] = char2id['BLANK']
        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']
            char_mask[i][j] = 1

        # for sentence pos
        tags = sen['sen_pos'].split()
        for j, pos in enumerate(tags):
            if j < max_length:
                if pos in pos2id:
                    sen_tag[i][j] = pos2id[pos]
                else:
                    sen_tag[i][j] = pos2id['UNK']
        for j in range(j + 1, max_length):
            sen_tag[i][j] = pos2id['BLANK']

        # MDP_word
        for j, path_word in enumerate(sen['MDP_w']):
            M_word = path_word.split()
            MDP_len[i][j] = min(len(M_word), MDP_length)
            for k, word in enumerate(M_word):
                if k < MDP_length:
                    if word in word2id:
                        MDP_word[i][j][k] = word2id[word]
                    else:
                        MDP_word[i][j][k] = word2id['UNK']
            for k in range(k + 1, MDP_length):
                MDP_word[i][j][k] = word2id['BLANK']
        for j in range(j + 1, path_num):
            MDP_word[i][j] = word2id['BLANK']

        # MDP_pos
        for j, path_pos in enumerate(sen['MDP_p']):
            M_pos = path_pos.split()
            for k, pos in enumerate(M_pos):
                if k < MDP_length:
                    if pos in pos2id:
                        MDP_pos[i][j][k] = pos2id[pos]
                    else:
                        MDP_pos[i][j][k] = pos2id['UNK']
            for k in range(k + 1, MDP_length):
                MDP_pos[i][j][k] = pos2id['BLANK']
        for j in range(j + 1, path_num):
            MDP_pos[i][j] = pos2id['BLANK']

        # MDP_rel
        for j, path_rel in enumerate(sen['MDP_r']):
            M_rel = path_rel.split()
            for k, rel in enumerate(M_rel):
                if k < MDP_length:
                    if rel in deprel2id:
                        MDP_rel[i][j][k] = deprel2id[rel]
                    else:
                        MDP_rel[i][j][k] = deprel2id['UNK']
            for k in range(k + 1, MDP_length):
                MDP_rel[i][j][k] = deprel2id['BLANK']
        for j in range(j + 1, path_num):
            MDP_rel[i][j] = deprel2id['BLANK']
        # MDP_dir

        for j, path_dir in enumerate(sen['MDP_d']):
            M_dir = path_dir.split()
            # print(M_dir)
            for k, dir in enumerate(M_dir):
                if k < MDP_length:
                    # print(dir)
                    MDP_dir[i][j][k] = int(dir)
            for k in range(k + 1, MDP_length):
                MDP_rel[i][j][k] = 3
        for j in range(j + 1, path_num):
            MDP_rel[i][j] = 4

        if sen['head']['word'] in word2id:
            head_ent[i] = word2id[sen['head']['word']]
        else:
            head_ent[i] = word2id['UNK']

        if sen['tail']['word'] in word2id:
            tail_ent[i] = word2id[sen['tail']['word']]
        else:
            tail_ent[i] = word2id['UNK']

        if sen['Root'] in word2id:
            root_verb[i] = word2id[sen['Root']]
        else:
            root_verb[i] = word2id['UNK']

        pos1, pos2 = find_pos(sen['sentence'], sen['head']['word'], sen['tail']['word'])
        if pos1 == -1 or pos2 == -1:
            raise Exception(
                "[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sen['sentence'],
                                                                                                 sen['head']['word'],
                                                                                                 sen['tail']['word']))
        if pos1 >= max_length:
            pos1 = max_length - 1
        if pos2 >= max_length:
            pos2 = max_length - 1
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for j in range(max_length):
            # sen_pos1, sen_pos2
            sen_pos1[i][j] = j - pos1 + max_length
            sen_pos2[i][j] = j - pos2 + max_length
            # sen_mask
            if j >= sen_len[i]:
                sen_mask[i][j] = [0, 0, 0]
            elif j - pos_min <= 0:
                sen_mask[i][j] = [100, 0, 0]
            elif j - pos_max <= 0:
                sen_mask[i][j] = [0, 100, 0]
            else:
                sen_mask[i][j] = [0, 0, 100]
        # bag_scope
        if is_training:
            tup = (sen['head']['id'], sen['tail']['id'], sen['relation'])
        else:
            tup = (sen['head']['id'], sen['tail']['id'])
        if bag_key == [] or bag_key[len(bag_key) - 1] != tup:
            bag_key.append(tup)
            bag_scope.append([i, i])
        bag_scope[len(bag_scope) - 1][1] = i

    print("Processing bag label...")
    # bag_label
    if is_training:
        for i in bag_scope:
            bag_label.append(sen_label[i[0]])
    else:
        for i in bag_scope:
            multi_hot = np.zeros(len(rel2id), dtype=np.int64)
            for j in range(i[0], i[1] + 1):
                multi_hot[sen_label[j]] = 1
            bag_label.append(multi_hot)
    print("Finish processing")
    # ins_scope
    ins_scope = np.stack([list(range(len(ori_data))), list(range(len(ori_data)))], axis=1)
    print("Processing instance label...")
    # ins_label
    if is_training:
        ins_label = sen_label
    else:
        ins_label = []
        for i in sen_label:
            one_hot = np.zeros(len(rel2id), dtype=np.int64)
            one_hot[i] = 1
            ins_label.append(one_hot)
        ins_label = np.array(ins_label, dtype=np.int64)
    print("Finishing processing")
    bag_scope = np.array(bag_scope, dtype=np.int64)
    bag_label = np.array(bag_label, dtype=np.int64)
    ins_scope = np.array(ins_scope, dtype=np.int64)
    ins_label = np.array(ins_label, dtype=np.int64)

    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "test"
    np.save(os.path.join(out_path, 'vec.npy'), word_vec_mat)
    np.save(os.path.join(out_path, 'char_vec.npy'), char_vec_mat)
    np.save(os.path.join(out_path, 'deprel_vec.npy'), deprel_vec_mat)
    np.save(os.path.join(out_path, name_prefix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + '_char.npy'), sen_char)
    np.save(os.path.join(out_path, name_prefix + '_sen_length.npy'), sen_len)
    np.save(os.path.join(out_path, name_prefix + '_char_mask.npy'), char_mask)
    np.save(os.path.join(out_path, name_prefix + '_pos1.npy'), sen_pos1)
    np.save(os.path.join(out_path, name_prefix + '_pos2.npy'), sen_pos2)
    np.save(os.path.join(out_path, name_prefix + '_tag.npy'), sen_tag)
    np.save(os.path.join(out_path, name_prefix + '_mask.npy'), sen_mask)
    np.save(os.path.join(out_path, name_prefix + '_MDPword.npy'), MDP_word)
    np.save(os.path.join(out_path, name_prefix + '_MDPrel.npy'), MDP_rel)
    np.save(os.path.join(out_path, name_prefix + '_MDPtag.npy'), MDP_pos)
    np.save(os.path.join(out_path, name_prefix + '_MDPdir.npy'), MDP_dir)
    np.save(os.path.join(out_path, name_prefix + '_MDP_length.npy'), MDP_len)
    np.save(os.path.join(out_path, name_prefix + '_head.npy'), head_ent)
    np.save(os.path.join(out_path, name_prefix + '_tail.npy'), tail_ent)
    np.save(os.path.join(out_path, name_prefix + '_root.npy'), root_verb)
    np.save(os.path.join(out_path, name_prefix + '_bag_label.npy'), bag_label)
    np.save(os.path.join(out_path, name_prefix + '_bag_scope.npy'), bag_scope)
    np.save(os.path.join(out_path, name_prefix + '_ins_label.npy'), ins_label)
    np.save(os.path.join(out_path, name_prefix + '_ins_scope.npy'), ins_scope)
    print("Finish saving")


init(train_file_name, word_file_name, char_file_name, rel_file_name, pos_file_name, deprel_file_name, max_length=80,
     MDP_length=10, case_sensitive=False, is_training=True)
init(test_file_name, word_file_name, char_file_name, rel_file_name, pos_file_name, deprel_file_name, max_length=80,
     MDP_length=10, case_sensitive=False, is_training=False)
