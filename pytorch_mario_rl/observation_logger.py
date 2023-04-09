import numpy as np
# import time, datetime
from scipy.io import savemat

class observation_logger():

    def __init__(self, save_dir):
        self.save_file = None
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True)

        # observation data
        self.actions = []
        self.states = []
        # self.data_log = []
        # self.info = []
        # self.width = []
        # self.height = []

        # Timing
        # self.record_time = time.time()


    def init_episode(self, episode):
        self.save_file = self.save_dir / f"{'mario_obs_log_'}{episode:04d}.csv"

        # observation data
        self.actions = []
        self.states = []
        # self.data_log = []
        # self.info = []


    def log_step(self, action, state):

        # step = np.array([action, h, w])
        # step = np.append(step, np.ravel(state))
        # step = np.append(step, state.flatten())
        # self.data_log.append(step.astype(np.uint8))

        self.actions.append(action)
        self.states.append(state.flatten().astype(np.uint8))


    def save(self, info, height, width):

        # np.save(self.save_file, np.array(self.data_log), allow_pickle=False, fix_imports=True)
        # np.savetxt(self.save_file, np.array(self.data_log), fmt='%d', delimiter=",")
        # self.info = info

        save_dict = {'states': np.array(self.states), 'actions': np.array(self.actions), 'height': height, 'width': width}
        save_dict.update(info)
        savemat(self.save_file.with_suffix(".mat"), save_dict)
