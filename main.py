from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Embedding, TimeDistributed, RepeatVector, Merge, LSTM
from keras.preprocessing import sequence
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

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPoolinga2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
#Initialize pre-trained weights
image_model = VGG_16('vgg16.h5')

image_model.layers.pop()
for layer in image_model.layers:
    layer.trainable = False

language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(LSTM(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))

# let's repeat the image vector to turn it into a sequence.
print("Repeat model loading")
image_model.add(RepeatVector(max_caption_len))
print("Repeat model loaded")
# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
print("Merging")
model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(LSTM(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print("Merged")

model.fit([images, captions], next_words, batch_size=1, epochs=5)

model.save_weights('image_caption_weights.h5')