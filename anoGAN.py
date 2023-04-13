from keras.utils.generic_utils import Progbar
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, \
    Flatten
from keras.layers.core import Activation
from keras.optimizers import Adam
import keras.backend as K
import math, cv2
import numpy as np
import os
from natsort import natsorted


# Generator
class Generator(object):
    def __init__(self, input_dim, image_shape):
        INITIAL_CHANNELS = 128
        INITIAL_SIZE = 128

        inputs = Input((input_dim,))
        fc1 = Dense(input_dim=input_dim, units=INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE)(inputs)
        fc1 = BatchNormalization()(fc1)
        fc1 = LeakyReLU(0.2)(fc1)
        fc2 = Reshape((INITIAL_SIZE, INITIAL_SIZE, INITIAL_CHANNELS),
                      input_shape=(INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE,))(fc1)
        up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
        conv1 = Conv2D(64, (3, 3), padding='same')(up1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(image_shape[2], (5, 5), padding='same')(up2)
        outputs = Activation('tanh')(conv2)

        self.model = Model(inputs=[inputs], outputs=[outputs])

    def get_model(self):
        return self.model


# Discriminator
class Discriminator(object):
    def __init__(self, input_shape):
        inputs = Input(input_shape)
        conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
        conv1 = LeakyReLU(0.2)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
        conv2 = LeakyReLU(0.2)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        fc1 = Flatten()(pool2)
        fc1 = Dense(1)(fc1)
        outputs = Activation('sigmoid')(fc1)

        self.model = Model(inputs=[inputs], outputs=[outputs])

    def get_model(self):
        return self.model


# DCGAN
class DCGAN(object):
    def __init__(self, input_dim, image_shape):
        self.input_dim = input_dim
        self.d = Discriminator(image_shape).get_model()
        self.g = Generator(input_dim, image_shape).get_model()

    def compile(self, g_optim, d_optim):
        self.d.trainable = False
        self.dcgan = Sequential([self.g, self.d])
        self.dcgan.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.d.trainable = True
        self.d.compile(loss='binary_crossentropy', optimizer=d_optim)

    def train(self, epochs, batch_size, X_train):
        g_losses = []
        d_losses = []
        for epoch in range(epochs):
            np.random.shuffle(X_train)
            n_iter = X_train.shape[0] // batch_size
            progress_bar = Progbar(target=n_iter)
            for index in range(n_iter):
                # create random noise -> N latent vectors
                noise = np.random.uniform(-1, 1, size=(batch_size, self.input_dim))

                # load real data & generate fake data
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                for i in range(batch_size):
                    if np.random.random() > 0.5:
                        image_batch[i] = np.fliplr(image_batch[i])
                    if np.random.random() > 0.5:
                        image_batch[i] = np.flipud(image_batch[i])
                generated_images = self.g.predict(noise, verbose=0)

                # attach label for training discriminator
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)

                # training discriminator
                d_loss = self.d.train_on_batch(X, y)

                # training generator
                g_loss = self.dcgan.train_on_batch(noise, np.array([1] * batch_size))

                progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            if (epoch + 1) % 10 == 0:
                image = self.combine_images(generated_images)
                image = (image + 1) / 2.0 * 255.0
                cv2.imwrite('/net/nfs2/export/home/tanaka/PycharmProjects/pythonProject/result/' + str(epoch) + ".png", image)
            print('\nEpoch' + str(epoch) + " end")

            # save weights for each epoch
            if (epoch + 1) % 10 == 0:
                self.g.save_weights('weights/generator_' + str(epoch) + '.h5', True)
                self.d.save_weights('weights/discriminator_' + str(epoch) + '.h5', True)
        return g_losses, d_losses

    def load_weights(self, g_weight, d_weight):
        self.g.load_weights(g_weight)
        self.d.load_weights(d_weight)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:4]
        image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
        return image


# AnoGAN
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))


class ANOGAN(object):
    def __init__(self, input_dim, g):
        self.input_dim = input_dim
        self.g = g
        g.trainable = False
        # Input layer cann't be trained. Add new layer as same size & same distribution
        anogan_in = Input(shape=(input_dim,))
        g_in = Dense((input_dim), activation='tanh', trainable=True)(anogan_in)
        g_out = g(g_in)
        self.model = Model(inputs=anogan_in, outputs=g_out)
        self.model_weight = None

    def compile(self, optim):
        self.model.compile(loss=sum_of_residual, optimizer=optim)
        K.set_learning_phase(0)

    def compute_anomaly_score(self, x, iterations=300):
        z = np.random.uniform(-1, 1, size=(1, self.input_dim))

        # learning for changing latent
        loss = self.model.fit(z, x, batch_size=1, epochs=iterations, verbose=0)
        loss = loss.history['loss'][-1]
        similar_data = self.model.predict_on_batch(z)

        return loss, similar_data


# train
if __name__ == '__main__':
    batch_size = 16
    epochs = 100
    input_dim = 30
    g_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)
    d_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)

    """"""
    #
    from torch.utils.data import Dataset
    data_list = []
    orn_list =[]
    path = '/net/nfs2/export/home/tanaka/data/normal_0.4'
    dir = os.listdir(path)
    dir = natsorted(dir)
    for p in dir[:540]:
        dir1 = os.listdir(path + '/' + p)
        dir1 = natsorted(dir1)
        data_list.append(dir1)   # ファイル名のリストを用意する

    path = '/net/nfs2/export/home/tanaka/data/orn_0.4'
    dir = os.listdir(path)
    dir = natsorted(dir)
    for p in dir[:]:
        dir1 = os.listdir(path + '/' + p)
        dir1 = natsorted(dir1)
        orn_list.append(dir1)

    class NpyDataset(Dataset):
        def __init__(self, file_list):
            self.file_list = file_list

        def __getitem__(self, index):
            file_name = self.file_list[index]
            data = np.load(file_name)
            return data.transpose(1, 2, 0)

        def __len__(self):
            return len(self.file_list)


    from sklearn.model_selection import KFold

    kf = KFold(n_splits=9, shuffle=False)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        # train_idx: 訓練データのインデックスのリスト
        # val_idx: バリデーションデータのインデックスのリスト

        # データセットを分割する
        train_file_list = [file_list[idx] for idx in train_idx]
        val_file_list = [file_list[idx] for idx in val_idx]

        # 訓練データとバリデーションデータのデータローダーを作成する
        train_dataset = NpyDataset(train_file_list)
        X_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = NpyDataset(val_file_list)
        X_test = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # モデルを学習する
        for i, data in enumerate(X_train):
        # 訓練データを使ってモデルを学習する処理をここに記述する

        # バリデーションデータを使ってモデルの性能を評価する処理をここに記述する

        input_shape = X_train[0].shape
    # train generator & discriminator
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.compile(g_optim, d_optim)
    g_losses, d_losses = dcgan.train(epochs, batch_size, X_train)
    with open('loss.csv', 'w') as f:
        for g_loss, d_loss in zip(g_losses, d_losses):
            f.write(str(g_loss) + ',' + str(d_loss) + '\n')

# test
K.set_learning_phase(1)


def denormalize(X):
    return ((X + 1.0) / 2.0 * 255.0).astype(dtype=np.uint8)


if __name__ == '__main__':
    iterations = 100
    input_dim = 30
    anogan_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # load weights
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.load_weights('weights/generator_99.h5', 'weights/discriminator_99.h5')  # 3999.99

    for i, test_img in enumerate(X_test):
        test_img = test_img[np.newaxis, :, :, :]
        anogan = ANOGAN(input_dim, dcgan.g)
        anogan.compile(anogan_optim)
        anomaly_score, generated_img = anogan.compute_anomaly_score(test_img, iterations)

        generated_img = denormalize(generated_img)
        imgs = np.concatenate((denormalize(test_img[0]), generated_img[0]), axis=1)
        cv2.imwrite('/net/nfs2/export/home/tanaka/PycharmProjects/pythonProject/predict/' + os.sep + str(int(anomaly_score)) + '_' + str(i) + '.png', imgs)
        print(str(i) + ' %.2f' % anomaly_score)
        with open('scores.txt', 'a') as f:
            f.write(str(anomaly_score) + '\n')
        if y_test[i] == 0:
            with open('scores_0.txt', 'a') as f:
                f.write(str(anomaly_score) + '\n')
        else:
            with open('scores_1.txt', 'a') as f:
                f.write(str(anomaly_score) + '\n')
                # plot histgram
    import matplotlib.pyplot as plt
    import csv

    x = []
    with open('scores_0.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = int(float(row[0]))
            x.append(row)
    y = []
    with open('scores_1.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = int(float(row[0]))
            y.append(row)

    plt.title("Histgram of Score")
    plt.xlabel("Score")
    plt.ylabel("freq")
    plt.hist(x, bins=40, alpha=0.3, histtype='stepfilled', color='r', label="1")
    plt.hist(y, bins=40, alpha=0.3, histtype='stepfilled', color='b', label='9')
    plt.legend(loc=1)
    plt.savefig("histgram.png")
    plt.show()
    plt.close()
