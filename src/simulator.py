import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RealtimeSystem:

    def __init__(self,
                 noise_function,
                 rest_amplitude,
                 rest_interval,
                 activity_amplitude,
                 activity_interval,
                 samples_per_minute):
        """
            noise_function: Noise Function
            rest_amplitude: Rest Amplitude
            rest_interval: Rest Interval (minutes)
            activity_amplitude: Activity Amplitude
            activity_interval: Activity Interval (minutes)
            samples_per_minute: Samples per Minute
        """
        self.noise_function = noise_function
        self.rest_interval = rest_interval
        self.activity_interval = activity_interval
        self.rest_amplitude = rest_amplitude
        self.activity_amplitude = activity_amplitude
        self.samples_per_minute = samples_per_minute
        self.samples = []

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
        return [self.rest_amplitude for k in range(self.samples_per_minute * self.rest_interval)]

    def construct_activity_samples(self):
        return [self.activity_amplitude for k in range(self.samples_per_minute * self.activity_interval)]


if __name__ == "__main__":
    rlts = RealtimeSystem(np.random.normal, 5, 15, 20, 60, 5)
    time = rlts.make_samples_from_feed()
    sns.lineplot(x=time, y=rlts.samples)
    plt.show()
