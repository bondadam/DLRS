import json
from typing import Any, List, TypeVar, Callable, Type, cast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


T = TypeVar("T")


def from_str(x: Any) -> str:
    return x


def from_int(x: Any) -> int:
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    return cast(Any, x).to_dict()


class Noise:
    type: str
    amplitude: int

    def __init__(self, type: str, amplitude: int) -> None:
        self.type = type
        self.amplitude = amplitude

    def apply(self, x):
        if self.type.lower() == "gaussian":
            return np.random.normal() * self.amplitude + x
        return x

    @staticmethod
    def from_dict(obj: Any) -> 'Noise':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        amplitude = from_int(obj.get("amplitude"))
        return Noise(type, amplitude)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["amplitude"] = from_int(self.amplitude)
        return result


class Impulsion:
    type: str
    fire_time: int
    duration: int

    def __init__(self, type: str, fire_time: int, duration: int) -> None:
        self.type = type
        self.fire_time = fire_time
        self.duration = duration

    @staticmethod
    def from_dict(obj: Any) -> 'Impulsion':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        fire_time = from_int(obj.get("fire_time"))
        duration = from_int(obj.get("duration"))
        return Impulsion(type, fire_time, duration)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["fire_time"] = from_int(self.fire_time)
        result["duration"] = from_int(self.duration)
        return result


class State:
    state: str
    duration: int
    amplitude: int
    impulsions: List[Impulsion]

    def __init__(self, state: str, duration: int, amplitude: int, impulsions: List[Impulsion]) -> None:
        self.state = state
        self.duration = duration
        self.amplitude = amplitude
        self.impulsions = impulsions

    def get_samples(self, dt_per_sample):
        samples_count = self.duration // dt_per_sample
        Y = np.full((samples_count,), self.amplitude)
        return Y

    @staticmethod
    def from_dict(obj: Any) -> 'State':
        assert isinstance(obj, dict)
        state = from_str(obj.get("state"))
        duration = from_int(obj.get("duration"))
        amplitude = from_int(obj.get("amplitude"))
        impulsions = from_list(Impulsion.from_dict, obj.get("impulsions"))
        return State(state, duration, amplitude, impulsions)

    def to_dict(self) -> dict:
        result: dict = {}
        result["state"] = from_str(self.state)
        result["duration"] = from_int(self.duration)
        result["amplitude"] = from_int(self.amplitude)
        result["impulsions"] = from_list(
            lambda x: to_class(Impulsion, x), self.impulsions)
        return result


class TransitionType:
    """
        Defines the Transition types betweens States.
        The change in amplitudes should not be Linear all the time.
        There are a number of easing types (i.e. Quadratic, Cubic, 
        Sine, Exponential, Elastic...) that contributes to the "realism"
        of the system, as all isn't "perfectly" Linear in the real world.

        type: str
            One of the predefined easing function.

    """
    type: str

    def __init__(self, type: str) -> None:
        self.type = type

    @staticmethod
    def from_dict(obj: Any) -> 'TransitionType':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        return TransitionType(type)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        return result


class RealtimeSystem:
    """
        realtime_tick: int
            Ticks in <milliseconds> before registering a sample and adding
            `dt` to the simulation's time.

        dt_per_sample: int
            Simulated time step between each sample in the simulation.
            This variable will likely be in <seconds>.

        transition_type: TransitionType
            The TransitionType between states. (e.g. EaseInOutQuad... etc.)

        noise: Noise
            The general Noise that is present for the entire system's samples.
            This simulates general noise such as Temperature, Humidity  as well
            as other factors that interfer with the sensors' readings.

        states: List[State]
            The list of States that this system emulates. 
    """
    realtime_tick: int
    dt_per_sample: int
    transition_type: TransitionType
    noise: Noise
    states: List[State]

    def __init__(self, realtime_tick: int, dt_per_sample: int, transition_type: TransitionType, noise: Noise, states: List[State]) -> None:
        self.realtime_tick = realtime_tick
        self.dt_per_sample = dt_per_sample
        self.transition_type = transition_type
        self.noise = noise
        self.states = states

    def total_time(self):
        """
            The sum of State durations in <seconds>.
        """
        return sum(map(lambda s: s.duration, self.states))

    def data(self):
        X = np.arange(0, self.total_time(), self.dt_per_sample)
        Y = np.concatenate([s.get_samples(self.dt_per_sample) for s in self.states], axis=None)
        Y = np.vectorize(self.noise.apply)(Y)
        return X, Y
    
    def render(self, color='red'):
        X, Y = rlts.data()
        ax = sns.lineplot(X, Y, color=color)
        ax.set_title("Realtime System")
        L = ax.lines[-1]
        x1 = L.get_xydata()[:,0]
        y1 = L.get_xydata()[:,1]
        ax.fill_between(x1,y1, color=color, alpha=0.2)

    @staticmethod
    def from_dict(obj: Any) -> 'RealtimeSystem':
        assert isinstance(obj, dict)
        realtime_tick = from_int(obj.get("realtime_tick"))
        dt_per_sample = from_int(obj.get("dt_per_sample"))
        transition_type = TransitionType.from_dict(obj.get("transition_type"))
        noise = Noise.from_dict(obj.get("noise"))
        states = from_list(State.from_dict, obj.get("states"))
        return RealtimeSystem(realtime_tick, dt_per_sample, transition_type, noise, states)

    def to_dict(self) -> dict:
        result: dict = {}
        result["realtime_tick"] = from_int(self.realtime_tick)
        result["dt_per_sample"] = from_int(self.dt_per_sample)
        result["transition_type"] = to_class(
            TransitionType, self.transition_type)
        result["noise"] = to_class(Noise, self.noise)
        result["states"] = from_list(lambda x: to_class(State, x), self.states)
        return result


if __name__ == "__main__":
    realtime_tick = 200 # milliseconds, (Animation: TODO)
    delta_time_per_sample = 20  # seconds
    transition = TransitionType("")  # No transitions, (First transition: TODO)
    noise = Noise("gaussian", 0.5)  # Normal function (Gaussian noise)
    # 3 states, no impulses (Impulses: TODO)
    states = [State("Rest", 1200, 5, []), State("Active", 3600, 20, []), State("Rest", 1200, 5, [])]

    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          transition,
                          noise,
                          states)
    rlts.render(color='blue')

    states = [State("Rest", 1200, 5, []), State("Active", 3600, 10, []), State("Rest", 1200, 5, [])]
    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          transition,
                          noise,
                          states)
    rlts.render(color='red')
    plt.show()
    print(rlts.to_dict())