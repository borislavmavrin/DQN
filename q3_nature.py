import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful

        # NOTE:
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param
              make sure to flatten() the tensor before connecting it to fully connected layers 

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################
        # pad to the size used in the paper: 84 x 84
        with tf.variable_scope(scope, reuse=reuse):
            # batch_padded = tf.pad(state, [[0, 0], [2, 2], [2, 2], [0, 0]])
            layer1 = tf.contrib.layers.conv2d(
                inputs=state,
                num_outputs=32,
                kernel_size=[8, 8],
                stride=4,
                padding="SAME",
                activation_fn=tf.nn.relu
                )
            layer2 = tf.contrib.layers.conv2d(
                inputs=layer1,
                num_outputs=64,
                kernel_size=[4, 4],
                stride=2,
                padding="SAME",
                activation_fn=tf.nn.relu
            )

            layer3 = tf.contrib.layers.conv2d(
                inputs=layer2,
                num_outputs=64,
                kernel_size=[3, 3],
                stride=1,
                padding="SAME",
                activation_fn=tf.nn.relu
            )
            layer4 = tf.contrib.layers.fully_connected(
                tf.reshape(layer3, [-1, 10 * 10 * 64]),
                512,
                activation_fn=tf.nn.relu
            )
            layer5 = tf.contrib.layers.fully_connected(
                layer4,
                num_actions,
                activation_fn=None
            )
            out = layer5

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
