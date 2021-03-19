from keras.layers import Lambda, Input, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VA_Model:
    def __init__(self):
        pass

    def get_dataset(self, state_num=10, time_len=50000, signal_dimension=15, CNR=1, window_len=11, half_window_len=5):

        self.state_num = state_num
        self.time_len = time_len
        self.signal_dimension = signal_dimension
        self.CNR = CNR
        self.window_len = window_len
        self.half_window_len = half_window_len
        a = np.ones(shape=(state_num, state_num))
        alpha = np.ones(10)*10
        alpha[5:] = 1
        base_prob = np.random.dirichlet(alpha) * 0.1
        for t in range(state_num):
            a[t, :] = base_prob
            a[t, t] += 0.9

        # simulate states
        state = np.zeros(time_len, dtype=np.uint8)
        p = np.random.uniform()
        state[0] = np.floor(p*state_num)
        for t in range(0, time_len-1):
            p = np.random.uniform()
            for s in range(state_num):
                if (p <= np.sum(a[state[t], :s+1])):
                    state[t+1] = s
                    break

        freq = np.zeros(state_num)
        for t in range(state_num):
            freq[t] = np.sum(state == t)
        loading = np.random.randint(-1, 2, size=(state_num, signal_dimension))

        cov = np.zeros((state_num, signal_dimension, signal_dimension))
        for t in range(state_num):
            cov[t, :, :] = np.matmul(np.transpose(
                [loading[t, :]]), [loading[t, :]])

        # generate BOLD signal
        signal = np.zeros((time_len, signal_dimension))
        for t in range(0, time_len):
            signal[t, :] = np.random.multivariate_normal(
                np.zeros((signal_dimension)), cov[state[t], :, :])
        signal += np.random.normal(size=signal.shape)/CNR
        original_dim = np.uint32(signal_dimension*(signal_dimension-1)/2)

        x_train = np.zeros(
            shape=(time_len-window_len*2, np.uint32(original_dim)))
        sum_corr = np.zeros(shape=(state_num, original_dim))
        occupancy = np.zeros(state_num)

        for t in range(window_len, time_len-window_len):
            corr_matrix = np.corrcoef(np.transpose(
                signal[t-half_window_len:t+half_window_len+1, :]))
            upper = corr_matrix[np.triu_indices(signal_dimension, k=1)]
            x_train[t-window_len, :] = np.squeeze(upper)
            if (np.sum(state[t-half_window_len:t+half_window_len+1] == state[t]) == window_len):
                sum_corr[state[t], :] += x_train[t-window_len, :]
                occupancy[state[t]] += 1

        self.x_train = x_train
        self.original_dim = original_dim

    def build_model(self):
        input_shape = (self.original_dim, )
        intermediate_dim = 64
        intermediate_dim_2 = 16
        latent_dim = 3
        cat_dim = 1
        class_num = 5
        self.class_num = class_num
        inputs = Input(shape=input_shape, name='encoder_input')
        inter_x1 = Dense(intermediate_dim, activation='tanh',
                         name='encoder_intermediate')(inputs)
        inter_x2 = Dense(intermediate_dim_2, activation='tanh',
                         name='encoder_intermediate_2')(inter_x1)
        inter_x3 = Dense(intermediate_dim_2, activation='tanh',
                         name='encoder_intermediate_3')(inter_x1)

        # add 3 means as additional parameters
        dummy = Input(shape=(1,), name='dummy')
        mu_vector = Dense(class_num*latent_dim, name='mu_vector',
                          use_bias=False)(dummy)
        mu = Reshape((class_num, latent_dim), name='mu')(mu_vector)

        # prior categorical distribution
        pi = Dense(class_num, activation='softmax', name='pi')(dummy)

        # posterior categorical distribution
        c = Dense(class_num, activation='softmax', name='c')(inter_x2)

        # outlier/non-outlier classification (Posterior Beta)
        # inter_outlier = Dense(128, activation='relu', name='inter_outlier')(x)
        c_outlier = Dense(2, activation='softmax', name='c_outlier')(inter_x3)

        # q(z|x)
        z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
        z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model([inputs, dummy], [z_mean, z_log_var, z,
                                               mu, c, c_outlier, pi], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        inter_y1 = Dense(intermediate_dim_2, activation='tanh')(latent_inputs)
        inter_y2 = Dense(intermediate_dim, activation='tanh')(inter_y1)
        outputs = Dense(self.original_dim, activation='tanh')(inter_y2)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = self.decoder(self.encoder([inputs, dummy])[2])
        self.model = Model([inputs, dummy], outputs, name='vae_mlp')
        self.inputs = inputs
        self.outputs = outputs
        self.c_outlier = c_outlier
        self.c = c
        self.z_log_var = z_log_var
        self.z_mean = z_mean
        self.mu = mu
        self.pi = pi

    def fit(self, epochs, batch_size=256, Lambda1=1, Lambda2=200, Alpha=0.075):
        models = (self.encoder, self.decoder)
        dummy_train = np.ones((self.x_train.shape[0], 1))
        self.dummy_train = dummy_train
        reconstruction_loss = mse(self.inputs, self.outputs)
        reconstruction_loss = K.tf.multiply(
            reconstruction_loss, self.c_outlier[:, 0])
        reconstruction_loss *= self.original_dim

        # sum over reconstruction loss and kl-div loss
        kl_loss_all = K.tf.get_variable("kl_loss_all", [batch_size, 1],
                                        dtype=K.tf.float32, initializer=K.tf.zeros_initializer)
        kl_cat_all = K.tf.get_variable("kl_cat_all", [batch_size, 1],
                                       dtype=K.tf.float32, initializer=K.tf.zeros_initializer)
        dir_prior_all = K.tf.get_variable("dir_prior_all", [batch_size, 1],
                                          dtype=K.tf.float32, initializer=K.tf.zeros_initializer)

        for i in range(0, self.class_num):
            # stick-breaking reconstruction of categorical distribution
            c_inlier = K.tf.multiply(self.c[:, i], self.c_outlier[:, 0])

            # kl-divergence between q(z|x) and p(z|c)
            kl_loss = 1 + self.z_log_var - \
                K.square(self.z_mean-self.mu[:, i, :]) - K.exp(self.z_log_var)
            kl_loss = K.tf.multiply(K.sum(kl_loss, axis=-1), c_inlier)
            kl_loss *= -0.5
            kl_loss_all += kl_loss

            # kl-divergence between q(c|x) and p(c) (not including outlier class)
            mc = K.mean(self.c[:, i])
            mpi = K.mean(self.pi[:, i])
            kl_cat = mc * K.log(mc) - mc * K.log(mpi)
            kl_cat_all += kl_cat

            # Dir prior: Dir(3, 3, ..., 3)
            dir_prior = -0.1*K.log(self.pi[:, i])
            dir_prior_all += dir_prior

        # kl-divergence between Beta prior and Beta posterior (outlier class)
        mco1 = K.mean(self.c_outlier[:, 0])
        mco2 = K.mean(self.c_outlier[:, 1])
        mpo1 = 1-Alpha
        mpo2 = Alpha
        kl_cat_outlier = (mco1 * K.log(mco1) - mco1 * np.log(mpo1) +
                          mco2 * K.log(mco2) - mco2 * np.log(mpo2))

        # total loss
        vae_loss = K.mean(reconstruction_loss +
                          kl_loss_all +
                          dir_prior_all +
                          Lambda1*kl_cat_all)+Lambda2*kl_cat_outlier

        self.model.add_loss(vae_loss)
        self.model.compile(optimizer='adam')
        self.model.summary()

        # vae.load_weights('vae_mlp_mnist.h5')
        self.model.fit([self.x_train, dummy_train],
                       epochs=epochs,
                       batch_size=batch_size)
        self.batch_size = batch_size

    def save(self):
        self.model.save_weights('vae_mlp_mnist.h5')

    def load(self, path):
        self.model.load_weights(path)

    def test(self):
        [z_mean, z_log_var, z, mu, c, c_outlier, pi] = self.encoder.predict(
            [self.x_train, self.dummy_train], batch_size=self.batch_size)

        # estimate label

        labels = np.zeros(self.x_train.shape[0])
        for i in range(0, self.x_train.shape[0]):
            max_prob = np.max(np.multiply(c[i, :], c_outlier[i, 0]))
            idx = np.argmax(np.multiply(c[i, :], c_outlier[i, 0]))
            if (max_prob > c_outlier[i, 1]):
                labels[i] = idx
            else:
                labels[i] = self.class_num
        return labels


model = VA_Model()
model.get_dataset()
model.build_model()
model.fit(1)
model.save()
model.test()