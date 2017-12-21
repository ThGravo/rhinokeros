import gym
from gym import Wrapper
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np


class AtariConvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        block4_pool_features = model.predict(x)


    @property
    def _step(self, action):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                _ = self.reset() # automatically reset the env
            done = True

        return observation, reward, done, info
