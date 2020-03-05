import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


class RealtimeSystem:

    def __init__(self,
                 noise_amplitude,
                 noise_function,
                 rest_amplitude,
                 rest_interval,
                 activity_amplitude,
                 activity_interval,
                 samples_per_minute):
        """
            noise_amplitude: Noise Amplitude
            noise_function: Noise Function
            rest_amplitude: Rest Amplitude
            rest_interval: Rest Interval (minutes)
            activity_amplitude: Activity Amplitude
            activity_interval: Activity Interval (minutes)
            samples_per_minute: Samples per Minute
        """
        self.noise_amplitude = noise_amplitude
        self.noise_function = noise_function
        self.rest_interval = rest_interval
        self.activity_interval = activity_interval
        self.rest_amplitude = rest_amplitude
        self.activity_amplitude = activity_amplitude
        self.samples_per_minute = samples_per_minute
        self.samples = []
        random.seed()

    @staticmethod
    def random_noise(base_amplitude, noise_amplitude):
        return base_amplitude + (random.uniform(-noise_amplitude, noise_amplitude))

    @staticmethod
    def gaussian_noise(base_amplitude, noise_amplitude):
        return base_amplitude + np.random.normal(loc=0, scale=noise_amplitude, size=1)[0]

    def total_samples(self):
        return self.samples_per_minute * (2 * self.rest_interval + self.activity_interval)

    def make_samples_from_feed(self, feed=["rest", "activity", "rest"]):
        self.samples = []
        for phase in feed:
            if phase.lower() == "rest":
                self.samples += self.construct_rest_samples()
            elif phase.lower() == "activity":
                self.samples += self.construct_activity_samples()
            else:
                raise Exception("Unknown state activity feed state.")
        return np.linspace(0, len(self.samples) / self.samples_per_minute, len(self.samples))

    def construct_rest_samples(self):
        return [self.noise_function(self.rest_amplitude, self.noise_amplitude) for k in range(self.samples_per_minute * self.rest_interval)]

    def construct_activity_samples(self):
        return [self.noise_function(self.activity_amplitude, self.noise_amplitude) for k in range(self.samples_per_minute * self.activity_interval)]


if __name__ == "__main__":
    rlts = RealtimeSystem(0.5,RealtimeSystem.gaussian_noise, 5, 15, 20, 60, 5)
    time = rlts.make_samples_from_feed()
    sns.lineplot(x=time, y=rlts.samples)
    plt.show()
