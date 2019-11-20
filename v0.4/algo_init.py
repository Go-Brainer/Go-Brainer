import tensorflow as tf
import h5py
import numpy as np
from multiprocessing import freeze_support
import time
import os
from keras.callbacks import ModelCheckpoint

from algo.data.parallel_processor import GoDataProcessor
from algo.agent.predict import DeepLearningAgent
from algo.networks.alphago import alphago_model
from algo.agent.pg import PolicyAgent
from algo.agent.predict import load_prediction_agent
from algo.encoders.alphago import AlphaGoEncoder
from algo.encoders.sevenplane import SevenPlaneEncoder
from algo.rl.simulate import experience_simulation
from algo.networks.alphago import alphago_model
from algo.rl import ValueAgent, load_experience
from algo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from algo.rl import load_value_agent
from algo.goboard_fast import GameState


def supervised_learning(board_size, pwd, filename, data_dir, samples, epochs):
    try:
        with tf.device("/GPU:0"):
            rows = cols = board_size
            num_classes = rows * cols

            #encoder = AlphaGoEncoder()
            encoder = SevenPlaneEncoder((board_size, board_size))
            processor = GoDataProcessor(encoder=encoder.name(), data_directory=data_dir)
            generator = processor.load_go_data('train', samples, use_generator=True)
            test_generator = processor.load_go_data('test', samples, use_generator=True)
            # end::alphago_sl_data[]

            # tag::alphago_sl_model[]
            input_shape = (encoder.num_planes, rows, cols)
            alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)
            alphago_sl_policy.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

            # bot_from_file = load_prediction_agent(h5py.File(pwd + "/" + filename, 'r'))
            # alphago_sl_policy = bot_from_file.model
            # end::alphago_sl_model[]

            # tag::alphago_sl_train[]
            batch_size = 128
            alphago_sl_policy.fit_generator(
                generator=generator.generate(batch_size, num_classes),
                epochs=epochs,
                steps_per_epoch=generator.get_num_samples() / batch_size,
                validation_data=test_generator.generate(batch_size, num_classes),
                validation_steps=test_generator.get_num_samples() / batch_size,
                callbacks=[ModelCheckpoint(pwd + "/checkpoints/" + filename.replace(".h5", "_{epoch}.h5"))],
                verbose=1
            )

            alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)
            with h5py.File(pwd + "/" + filename, 'w') as sl_agent_out:
                alphago_sl_agent.serialize(sl_agent_out)
            # end::alphago_sl_train[]

            alphago_sl_policy.evaluate_generator(
                generator=test_generator.generate(batch_size, num_classes),
                steps=test_generator.get_num_samples() / batch_size
            )
    except RuntimeError as e:
        print(e)

def reinforcement_learning():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_soft_device_placement(True)
        except RuntimeError as e:
            print(e)



def main():
    board_size = 19
    sl_samples = 10000
    sl_epochs = 200
    pwd = "./ag_agents"
    if not os.path.isdir(pwd):
        os.makedirs(pwd)

    sl_dir = pwd + "/sl"
    if not os.path.isdir(sl_dir):
        os.makedirs(sl_dir)
    kgs_dir = sl_dir + "/kgs_data"
    if not os.path.isdir(kgs_dir):
        os.makedirs(kgs_dir)
    sl_filename = "ag_sl_0.h5"
    supervised_learning(board_size, sl_dir, sl_filename, kgs_dir, sl_samples, sl_epochs)

#    reinforcement_learning()



if __name__ == '__main__':
    main()
