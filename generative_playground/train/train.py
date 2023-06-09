import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import argparse

class ImageLoader:
    def __init__(self, file_list, resize = None, return_fname = False):
        self.file_list = file_list
        self.resize = resize
    
    def __getitem__(self, i):
            img, filename = plt.imread(self.file_list[i]), self.file_list[i]
            if self.resize is not None:
                img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
            img = img / 255.0
            return img, filename
    
    def __len__(self):
        return len(self.file_list)

class ConditionalGAN(keras.Model):
    def __init__(self, critic, generator):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn_gen, loss_fn_critic):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn_gen = loss_fn_gen
        self.loss_fn_critic = loss_fn_critic

    def train_step(self, data):
        x, y = data
        
        # Train the critic.
        with tf.GradientTape() as tape:
            criticized_real = self.critic(x[y==1])
            criticized_fake = self.critic(self.generator(x[y==0]))
            d_loss = self.loss_fn_critic(criticized_real, criticized_fake)
        grads = tape.gradient(d_loss, self.critic.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.critic.trainable_weights)
        )

        # Train the generator.
        with tf.GradientTape() as tape:
            criticized_fake = self.critic(self.generator(x[y==0]))
            g_loss = self.loss_fn_gen(criticized_fake)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

def critic_loss(criticized_real, criticized_fake):
    return  (-1) * tf.math.reduce_mean(criticized_real) + tf.math.reduce_mean(criticized_fake)

def generator_loss(criticized_fake):
    return (-1) * tf.math.reduce_mean(criticized_fake)


def get_parser():
    pass

def train(args):
    metadata_frame = pd.read_csv("/Users/lukasadamek/Projects/GenerativePlayground/datasets/unzipped/metadata.csv")
    cleaned_frame = metadata_frame.query("(abs(shape_dim_zero - shape_dim_one)/shape_dim_one) < 0.15")
    cleaned_frame = cleaned_frame.query("(image_shape == 3) and (last_image_dimension == 3) and (shape_dim_zero >= 500) and (shape_dim_one >= 500)")

    generator = keras.models.Sequential()
    generator.add(keras.layers.Conv2D(3, (3, 3), padding="SAME", activation='relu', input_shape=(600, 600, 3)))
    generator.add(keras.layers.Conv2D(6, (5, 5), padding="SAME", activation='relu'))
    generator.add(keras.layers.Conv2D(9, (5, 5), padding="SAME", activation='relu'))
    generator.add(keras.layers.Conv2D(6, (5, 5), padding="SAME", activation='relu'))
    generator.add(keras.layers.Conv2D(3, (5, 5), padding="SAME", activation='relu'))
    generator.add(tf.keras.layers.Activation("sigmoid"))

    scorer = keras.models.Sequential()
    scorer.add(keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(600, 600, 3)))
    scorer.add(keras.layers.MaxPooling2D((4, 4)))
    scorer.add(keras.layers.Conv2D(3, (5, 5), activation='relu'))
    scorer.add(keras.layers.MaxPooling2D((4, 4)))
    scorer.add(keras.layers.Conv2D(1, (5, 5), activation='relu'))
    scorer.add(keras.layers.MaxPooling2D((4, 4)))
    scorer.add(keras.layers.Flatten())
    scorer.add(keras.layers.Dense(1, activation="linear"))

    cond_gan = ConditionalGAN(
        scorer, generator
    )

    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn_gen=generator_loss,
        loss_fn_critic=critic_loss,
    )

    cond_gan.fit(full_generator, epochs=1)