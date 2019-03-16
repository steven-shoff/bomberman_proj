import numpy as np
import pandas as pd


class QTableModel(object):

    def __init__(self, config):
        """
        Class initialization
        :param config: model parameters dictionary
        """

        self.name = 'Q-table'
        self.eps = float(config['eps'])
        self.alpha = float(config['alpha'])
        self.gamma = float(config['gamma'])
        self.qtable = None

    def load_table(self, load_path):
        self.qtable = pd.read_csv(load_path,
                          dtype={'0': np.float32, '1': np.float32, '2': np.float32, '3': np.float32, '4': np.float32,
                                 '5': np.float32}, index_col=0)
        self.qtable.index = self.qtable.index.astype(str)
        self.qtable.columns = self.qtable.columns.astype(int)

    def init_table(self):
        self.qtable = pd.DataFrame(np.zeros((4, 6)), index=['4000011010', '3000110010', '1010011000', '2010110000'])
        self.qtable.columns = self.qtable.columns.astype(int)

    def predict(self, state):
        """
        This class function predicts target
        :param state            : State in current round
        :return determined_state: determined state
        """
        if self.qtable is None:
            self.init_table()
        self.check_state(state)
        state_actions = self.qtable.loc[state].values
        return [state_actions]

    def check_state(self, state):
        """
        This class function adds unknown state into original Q-table
        :param state: State to be checked
        :return: None
        """
        if state not in self.qtable.index:
            self.qtable = self.qtable.append(pd.DataFrame(np.zeros((1, 6)), index=[state]))
            print('New state added: {}'.format(state))

    def fit(self, memory):  # Do update after each action in training
        """
        This class function updates Q-table by rewarding good actions and penalizing bad actions
        :param memory   : Memory in last move
        :return         : None
        """
        state, target, reward, state_new = memory
        self.check_state(state_new)
        old_q = self.qtable.loc[state, target]
        self.qtable.loc[state, target] = old_q + self.alpha * (reward + self.gamma * np.max(self.qtable.loc[state_new].values - old_q))

    def save(self, save_path):
        """
        This class functino saves Q-table in learning model
        """
        self.qtable.to_csv(save_path)
