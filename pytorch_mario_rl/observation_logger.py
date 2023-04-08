import numpy as np
import time, datetime

class observation_logger():

    def __init__(self, save_dir):
        self.save_file = None
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True)

        # self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        # self.ep_lengths_plot = save_dir / "length_plot.jpg"
        # self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        # self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # observation data
        # self.action = []
        # self.state = []
        self.data_log = []

        # # Moving averages, added for every call to record()
        # self.moving_avg_ep_rewards = []
        # self.moving_avg_ep_lengths = []
        # self.moving_avg_ep_avg_losses = []
        # self.moving_avg_ep_avg_qs = []

        # Current episode metric
        # self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, action, h, w, state):

        step = np.array([action, h, w])
        step = np.append(step, np.ravel(state))
        self.data_log.append(step.astype(np.uint8))


    def init_episode(self, episode):
        self.save_file = self.save_dir / f"{'mario_obs_log_'}{episode:04d}.csv"

        # observation data
        self.data_log = []
        # self.state = []

        # with open(self.save_file, "w") as f:
        #     f.write(
        #         # f"{'Time,':>20}{'action,':>8}{'State':<100}\n"
        #         f"{'Time, '}{'action, '}{'State '}\n"
        #     )

    def save(self):

        # self.state.append(state)
        # self.action.append(action)
        np.save(self.save_file, np.array(self.data_log), allow_pickle=False, fix_imports=True)

        np.savetxt(self.save_file, np.array(self.data_log), fmt='%d', delimiter=",")

        # last_record_time = self.record_time
        # self.record_time = time.time()
        # time_since_last_record = np.round(self.record_time - last_record_time, 3)

        # print(
        #     f"Episode {episode} - "
        #     f"Step {step} - "
        #     f"Epsilon {epsilon} - "
        #     f"Mean Reward {mean_ep_reward} - "
        #     f"Mean Length {mean_ep_length} - "
        #     f"Mean Loss {mean_ep_loss} - "
        #     f"Mean Q Value {mean_ep_q} - "
        #     f"Time Delta {time_since_last_record} - "
        #     f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        # )

        # with open(self.save_file, "a") as f:
        #     f.write(
        #         f"{time_since_last_record:15.3f}"
        #         f"{action}{','}{state}"
        #         # f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
        #         # f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
        #         f"\n"
        #     )

