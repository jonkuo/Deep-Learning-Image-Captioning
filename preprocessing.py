import cv2, json, numpy as np

#Load data
with open('annotations/captions_train2014.json') as data:
    data = json.load(data)
captions = data['annotations']
images = data['images']

#5 training examples
train = images
image_ids = [i['id'] for i in train]

#pre-process captions
train_indices = [next(index for (index, d) in enumerate(captions) if d["image_id"] == i) for i in image_ids]
train_captions = [captions[i]['caption'] for i in train_indices]
train_captions = ["START " + i.replace('.', '').lower() + " END" for i in train_captions]

#get filename for each image with corresponding captions
train_files = ["train2014/" + train[i]['file_name'] for i in range(len(train_captions))]
#pre-process image
images = []
for image in train_files:
    img = cv2.imread(image)
    img.resize((224,224,3))
    images.append(img)
images = np.asarray(images)
#build vocabulary and captions
words = [txt.split() for txt in train_captions]
unique = []
for word in words:
    unique.extend(word)
unique = list(set(unique))

# word_index = {}
# index_word = {}
index_word = {i: word for (i, word) in enumerate(unique)}
word_index = {word: i for (i, word) in enumerate(unique)}
# for i,word in enumerate(unique):
#     word_index[word] = i
#     index_word[i] = word

captions = [[word_index[txt] for txt in text.split()] for text in train_captions]
# for text in train_captions:
#     one = 
#     captions.append(one)

max_caption_len = max(len(i) for i in captions)
vocab_size = len(index_word)

captions = sequence.pad_sequences(captions, maxlen=max_caption_len,padding='post')
next_words = np.zeros((len(train_captions),vocab_size))
for i,text in enumerate(train_captions):
    text = text.split()
    x = [word_index[txt] for txt in text]
    x = np.asarray(x)
    next_words[i,x] = 1