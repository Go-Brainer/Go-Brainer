# tag::gtp_pachi[]
from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.agent.predict import load_prediction_agent
import h5py
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

bot = load_prediction_agent(h5py.File("./agents/deep_bot20191113-195136._s1000e20.h5", "r"))
gtp_bot = LocalGtpBot(go_bot=bot, termination=PassWhenOpponentPasses(),
                      handicap=9, opponent='gnugo')
gtp_bot.run()
# end::gtp_pachi[]
