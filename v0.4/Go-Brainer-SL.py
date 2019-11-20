# tag::e2e_imports[]
import h5py
import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder as XPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large
from multiprocessing import freeze_support
# end::e2e_imports[]


def main():
    samp = 1000
    epo = 15

    # tag::e2e_processor[]
    model_h5filename = "./agents/deep_bot_A.h5"

    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = XPlaneEncoder((go_board_rows, go_board_cols))
    data_dir = "data/" + str(encoder.num_planes) + "-planes"
    processor = GoDataProcessor(encoder=encoder.name(), data_directory=data_dir)

    for i in range(5):
        X, y = processor.load_go_data(num_samples=samp)
        # end::e2e_processor[]

        model_file = h5py.File(model_h5filename, "r")
        bot_from_file = load_prediction_agent(model_file)
        model = bot_from_file.model

        # tag::e2e_model[]
        try:
            with tf.device('/device:GPU:0'):
                model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

                model.fit(X, y, batch_size=128, epochs=epo, verbose=1)
                # end::e2e_model[]

                # tag::e2e_agent[]
                deep_learning_bot = DeepLearningAgent(model, encoder)
                deep_learning_bot.serialize(h5py.File(model_h5filename.replace('.h5', str(i)+'.h5'), "w"))
                # end::e2e_agent[]
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    freeze_support()
    main()

