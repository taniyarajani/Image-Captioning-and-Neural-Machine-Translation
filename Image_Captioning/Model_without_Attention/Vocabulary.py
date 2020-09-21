
import os
import json
import time
import numpy as np
import nltk
from PIL import Image
from shutil import copyfile
from collections import Counter


###################### Separting the files to train test and validation #########################

def read_captions(filepath):
    captions_dict = {}
    with open(filepath) as f:
        for line in f:
            line_split = line.split(sep='\t', maxsplit=1)
            caption = line_split[1][:-1]
            id_image = line_split[0].split(sep='#')[0]
            if id_image not in captions_dict:
                captions_dict[id_image] = [caption]
            else:
                captions_dict[id_image].append(caption)
    return captions_dict

def get_ids(filepath):
    ids = []
    with open(filepath) as f:
        for line in f:
            ids.append(line[:-1])
    return ids

def copyfiles(dir_output, dir_input, ids):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    for cur_id in ids:
        path_input = os.path.join(dir_input, cur_id)
        path_output = os.path.join(dir_output, cur_id)
        copyfile(path_input, path_output)

def write_captions(dir_output, ids, captions_dict):
    output_path = os.path.join(dir_output, 'captions.txt')
    output = []
    for cur_id in ids:
        cur_dict = {cur_id: captions_dict[cur_id]}
        output.append(json.dumps(cur_dict))

    with open(output_path, mode='w') as f:
        f.write('\n'.join(output))

def segregate(dir_images, filepath_token, captions_path_input):
    dir_output = {'train': 'train','dev'  : 'dev','test' : 'test'}

    # id [caption1, caption2, ..]
    captions_dict = read_captions(filepath_token)

    # train, dev, test images mixture
    images = os.listdir(dir_images)

    # read ids
    ids_train = get_ids(captions_path_input['train'])
    ids_dev = get_ids(captions_path_input['dev'])
    ids_test = get_ids(captions_path_input['test'])
    
    # copy images to respective dirs
    copyfiles(dir_output['train'], dir_images, ids_train)
    copyfiles(dir_output['dev'], dir_images, ids_dev)
    copyfiles(dir_output['test'], dir_images, ids_test)
    
    # write id
    write_captions(dir_output['train'], ids_train, captions_dict)
    write_captions(dir_output['dev'], ids_dev, captions_dict)
    write_captions(dir_output['test'], ids_test, captions_dict)

def load_captions(captions_dir):
    caption_file = os.path.join(captions_dir, 'captions.txt')
    captions_dict = {}
    with open(caption_file) as f:
        for line in f:
            cur_dict = json.loads(line)
            for k, v in cur_dict.items():
                captions_dict[k] = v
    return captions_dict

if __name__ == '__main__':
    dir_images = 'flickr8k/Flickr_Data/Flickr_Data/Images/'
    dir_text = 'flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/'
    filename_token = 'Flickr8k.token.txt'
    filename_train = 'Flickr_8k.trainImages.txt'
    filename_dev = 'Flickr_8k.devImages.txt'
    filename_test = 'Flickr_8k.testImages.txt'
    filepath_token = os.path.join(dir_text, filename_token)
    print(filepath_token)
    captions_path_input = {'train': os.path.join(dir_text, filename_train),'dev': os.path.join(dir_text, filename_dev),
                           'test': os.path.join(dir_text, filename_test)}
    tic = time.time()
    segregate(dir_images, filepath_token, captions_path_input)
    toc = time.time()
    print('time: %.2f mins' %((toc-tic)/60))


class Vocabulary():
    def __init__(self, captions_dict, threshold):
        self.word2id = {}
        self.id2word = {}
        self.index = 0
        self.build(captions_dict, threshold)

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.index
            self.id2word[self.index] = word
            self.index += 1

    def get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.word2id['<unk>']

    def get_word(self, index):
        return self.id2word[index]

    def build(self, captions_dict, threshold):
        counter = Counter()
        tokens = []
        for k, captions in captions_dict.items():
            for caption in captions:
                tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))

        counter.update(tokens)

        words = [word for word, count in counter.items() if count >= threshold]

        self.add_word('<unk>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<pad>')

        for word in words:
            self.add_word(word)

    def get_sentence(self, ids_list):
        sent = ''
        for cur_id in ids_list:
            cur_word = self.id2word[cur_id.item()]
            sent += ' ' + cur_word
            if cur_word == '<end>':
                break
        return sent

if __name__ == '__main__':

    captions_dict = load_captions('train')
    vocab = Vocabulary(captions_dict, 5)
    print(vocab.index)

