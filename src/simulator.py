import json
from typing import Any, List, TypeVar, Callable, Type, cast
from enum import Enum, auto
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
from random import randint, random, randrange
import os
from shutil import rmtree



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
    amplitude: int

    def __init__(self, type: str, fire_time: int, duration: int, amplitude: int) -> None:
        self.type = type
        self.fire_time = fire_time
        self.duration = duration
        self.amplitude = amplitude

    @staticmethod
    def from_dict(obj: Any) -> 'Impulsion':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        fire_time = from_int(obj.get("fire_time"))
        duration = from_int(obj.get("duration"))
        amplitude = from_int(obj.get("amplitude"))
        return Impulsion(type, fire_time, duration, amplitude)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["fire_time"] = from_int(self.fire_time)
        result["duration"] = from_int(self.duration)
        result["amplitude"] = from_int(amplitude)
        return result
    
    def apply(self, samples, dt_per_sample):
        """
            Applies an impulsion (large variation in amplitude) to the given array of samples
            starting at fire_time and ending after duration 
        """
        impulsion_start_sample = self.fire_time // dt_per_sample
        impulsion_end_sample = ((self.fire_time + self.duration) // dt_per_sample)
        ## TODO: check that the given values are correct (ie 0 < fire_time < state_total_time)
        affected_samples = samples[impulsion_start_sample : impulsion_end_sample]
        modified_samples = self._apply_func(affected_samples)
        all_samples = np.concatenate((samples[:impulsion_start_sample], modified_samples, samples[impulsion_end_sample:]))
        return all_samples


    def _apply_func(self, array):
        return self._type_as_func(array)

    def _type_as_func(self, t):
        impulsion_type = self.type.lower()
        if impulsion_type == "tooth":
            return self.tooth(t)
        else:
            return self.tooth(t)
    
    def tooth(self, t):
        if len(t) == 0: return t
        average_amplitude = sum(t)/len(t)
        linear_coefficient = ((average_amplitude - self.amplitude) * 2) / len(t)
        halfway_point = len(t)//2
        modified_array = []
        for i in range(len(t)):
            if i <= halfway_point:
                new_value = average_amplitude - (i * linear_coefficient)
            else:
                new_value = average_amplitude - (len(t) - i) * linear_coefficient
            modified_array.append(new_value) 
        return np.array(modified_array)


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

    def total_samples(self, dt_per_sample):
        """
            Total samples of the current state == State's Duration // System's Delta Time
        """
        return self.duration // dt_per_sample

    def get_samples(self, dt_per_sample):
        """
            Numpy array containing all the state's samples.
        """
        samples_count = self.total_samples(dt_per_sample)
        Y = np.full((samples_count,), self.amplitude)
        return Y
    
    def copy(self, impulsions):
        return State(self.state, self.duration, self.amplitude, impulsions)

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

class StateFactory:

    @staticmethod
    def periodic(n, states, with_random_impulses=False):
        newStates = []
        for i in range(n):
            newStates += states
        return newStates
    
    @staticmethod
    def random(n, rest: State, active: State, with_random_impulses=False, random_frequency=0.4, max_impulses_per_state=10):
        newStates = []
        states = [rest, active]
        for i in range(n):
            if with_random_impulses:
                time = lambda x, state: state.duration * x / max_impulses_per_state
                fire_time = lambda x, state: randrange(time(x, state), time(x+1, state))
                duration = lambda x, state: randint(0, abs(randint(1,2) * time(x + 1, state) - fire_time(x, state)))
                # 0% to 10% of state's amplitude
                amplitude = lambda state: random() * state.amplitude
                r, a = states
                rMax = randint(0, max_impulses_per_state)
                aMax = randint(0, max_impulses_per_state)
                if random() < random_frequency:
                    r = r.copy([Impulsion("tooth", fire_time(x, r), duration(x, r), amplitude(r)) for x in range(1, rMax+1) if random() < random_frequency])
                if random() < random_frequency:
                    a = a.copy([Impulsion("tooth", fire_time(x, a), duration(x, a), amplitude(a)) for x in range(1, aMax+1) if random() < random_frequency])
                states = [r, a]
            newStates.extend(states)
        return newStates


class TransitionType:
    """
        Defines the Transition types betweens States.
        The change in amplitudes should not be Linear all the time.
        There are a number of easing types (i.e. Quadratic, Cubic, 
        Sine, Exponential, Elastic...) that contributes to the "realism"
        of the system, as all isn't "perfectly" Linear in the real world.

        type: str
            One of the predefined easing function.
            Easing Functions Types: `quad`, `cubic`, `quartic`, `sine`, `circular`, `exponential`, `elastic`,
            `back`, `bounce`.

            These functions were fetched from: https://github.com/semitable/easing-functions
        
        samples_per_side: int
            How many samples to take from each State's array of samples endpoints.
            e.g:
                If we have 3 states and the array holding their samples is
                [[5,5,5,5,5,5], [20,20,20,20,20,20], [10,10,10,10,10,10]] and samples_per_side is 3
                then the Transition Function will interpolate the following arrays:

                [[5, 5, 5, 5, 5, 5], [20, 20, 20, 20, 20, 20], [10, 10, 10, 10, 10, 10]]
                           _____________________  ________________________
                
                => [[5, 5, 5, 20, 20, 20], [20, 20, 20, 10, 10, 10]]
    """
    type: str

    def __init__(self, type: str, samples_per_side: int) -> None:
        self.type = type
        self.samples_per_side = samples_per_side

    def apply(self, array_of_samples):
        """
            Applies a Transition Function between 2 States given an array `array_of_samples`
            which contain all samples from all States.
            If there are less than 2 States' Samples, the given array will be returned.
        """
        N = len(array_of_samples)
        for i in range(N - 1):
            ## The min function is to avoid giving a number of samples_per_side bigger than the total samples
            ## of the 2 states we're applying the easing function to.
            samples_per_side = np.min((self.samples_per_side, len(array_of_samples[i]), len(array_of_samples[i+1])))
            T = np.concatenate((array_of_samples[i][-samples_per_side:], array_of_samples[i + 1][:samples_per_side]), axis=None)
            T = self._apply_func(T)
            array_of_samples[i][- samples_per_side:] = T[: samples_per_side]
            array_of_samples[i + 1][:samples_per_side] = T[samples_per_side:]
        return array_of_samples


    def _apply_func(self, array):
        y_end = array[-1]
        y_start = array[0]
        t = np.arange(0, len(array) + 1) / len(array)
        t = t[1:]
        x = np.vectorize(self._type_as_func)(t)
        return y_end * x + y_start * (1 - x)

    def _type_as_func(self, t):
        ease = self.type.lower()
        if ease in ["quad"]:
            return TransitionType.quadEaseInOut(t)
        elif ease in ["cubic"]:
            return TransitionType.cubicEaseInOut(t)
        elif ease in ["quartic"]:
            return TransitionType.quarticEaseInOut(t)
        elif ease in ["sine"]:
            return TransitionType.sineEaseInOut(t)
        elif ease in ["circular"]:
            return TransitionType.circularEaseInOut(t)
        elif ease in ["elastic"]:
            return TransitionType.elasticEaseInOut(t)
        elif ease in ["exponential"]:
            return TransitionType.exponentialEaseInOut(t)
        elif ease in ["back"]:
            return TransitionType.backEaseInOut(t)
        elif ease in ["bounce"]:
            return TransitionType.bounceEaseInOut(t)
        else:
            return TransitionType.linearInOut(t)
    
    @staticmethod
    def linearInOut(t):
        return t

    ### Quadratic

    @staticmethod
    def quadEaseInOut(t):
        if t < 0.5:
            return 2 * t * t
        return (-2 * t * t) + (4 * t) - 1
        
    ### Cubic

    @staticmethod
    def cubicEaseInOut(t):
        if t < 0.5:
            return 4 * t * t * t
        p = 2 * t - 2
        return 0.5 * p * p * p + 1

    ### Quartic

    @staticmethod
    def quarticEaseInOut(t):
        if t < 0.5:
            return 8 * t * t * t * t
        p = t - 1
        return - 8 * p * p * p * p + 1
    
    ### Sine

    @staticmethod
    def sineEaseInOut(t):
        return 0.5 * (1 - np.cos(t * np.pi))

    ### Circular

    @staticmethod
    def circularEaseInOut(t):
        if t < 0.5:
            return 0.5 * (1 - np.sqrt(1 - 4 * (t * t)))
        return 0.5 * (np.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1)

    ### Exponential

    @staticmethod
    def exponentialEaseInOut(t):
        if t == 0 or t == 1:
            return t
        if t < 0.5:
            return 0.5 * np.pow(2, (20 * t) - 10)
        return - 0.5 * np.pow(2, (-20 * t) + 10) + 1
        
    ### Elastic

    @staticmethod
    def elasticEaseInOut(t):
        if t < 0.5:
            return (
                0.5
                * np.sin(13 * np.pi / 2 * (2 * t))
                * np.pow(2, 10 * ((2 * t) - 1))
            )
        return 0.5 * (
            np.sin(-13 * np.pi / 2 * ((2 * t - 1) + 1))
            * np.pow(2, -10 * (2 * t - 1))
            + 2
        )
    
    ### Back

    @staticmethod
    def backEaseInOut(t):
        if t < 0.5:
            p = 2 * t
            return 0.5 * (p * p * p - p * np.sin(p * np.pi))

        p = 1 - (2 * t - 1)

        return 0.5 * (1 - (p * p * p - p * np.sin(p * np.pi))) + 0.5

    ### Bounce


    @staticmethod
    def _bounceEaseIn(t):
        return 1 - TransitionType._bounceEaseOut(1 - t)


    @staticmethod
    def _bounceEaseOut(t):
        if t < 4 / 11:
            return 121 * t * t / 16
        elif t < 8 / 11:
            return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
        elif t < 9 / 10:
            return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
        return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0

    @staticmethod
    def bounceEaseInOut(t):
        if t < 0.5:
            return 0.5 * TransitionType._bounceEaseIn(t * 2)
        return 0.5 * TransitionType._bounceEaseOut(t * 2 - 1) + 0.5

    @staticmethod
    def from_dict(obj: Any) -> 'TransitionType':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        samples_per_side = from_str(obj.get("samples_per_side"))
        return TransitionType(type, samples_per_side)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["samples_per_side"] = from_int(self.samples_per_side)
        return result


class RealtimeSystem:
    """
        realtime_tick: int
            Ticks in <milliseconds> before registering a sample and adding
            `dt` to the simulation's time.

        dt_per_sample: int
            Simulated time step between each sample in the simulation.
            This variable will likely be in <seconds>.

        scale: int
            Amount of time to be displayed at one time on the graph.
            Example: 24hrs of state data, scale = 8hrs -> graph will show values 0 to 8*60*60
            Value is expected to be in seconds.
            Default value (No scaling) : 0

        offset: int
            Display values only after a certain time has elapsed.
            In other words, the current "position" in time. Can be combined with scale.
            Example: 24hrs of state data, offset = 3hrs, scale = 8hrs -> graph will show values from 3 * 60² to 8 * 60²
            Value is expected to be in seconds.
            Default value (No offset) : 0

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
    scale: int
    offset: int
    transition_type: TransitionType
    noise: Noise
    states: List[State]

    def __init__(self, realtime_tick: int, dt_per_sample: int, scale: int, offset: int, transition_type: TransitionType, noise: Noise, states: List[State]) -> None:
        self.realtime_tick = realtime_tick
        self.dt_per_sample = dt_per_sample
        self.scale = scale
        self.offset = offset
        self.transition_type = transition_type
        self.noise = noise
        self.states = states
        self.X = None
        self.Y = None

    def total_time(self):
        """
            The sum of State durations in <seconds>.
        """
        return sum(map(lambda s: s.duration, self.states))

    def data(self):
        X = np.arange(0, self.total_time(), self.dt_per_sample)
        Y = [s.get_samples(self.dt_per_sample) for s in self.states]
        ## Always apply impulses after transitions or the transitions will erase impulsions too close to
        ## the beginning or end of a state
        Y = self.transition_type.apply(Y)
        for i in range(len(self.states)):
            if len(self.states[i].impulsions) > 0:
                for impulsion in self.states[i].impulsions:
                    Y[i] = impulsion.apply(Y[i], self.dt_per_sample)
        Y = np.concatenate(Y, axis=None)
        Y = np.vectorize(self.noise.apply)(Y)
        self.X = X
        self.Y = Y
        return X, Y
    
    def data_scaled(self):
        if self.Y is None:
            _, Y = self.data()
        else:
            Y = self.Y
        X = np.arange(self.offset, self.offset + self.scale, self.dt_per_sample)
        start_index = self.offset // self.dt_per_sample
        end_index = start_index + self.scale // self.dt_per_sample
        Y = Y[start_index:end_index]
        return X, Y

    def generate_frames(self, folders, unique=False):
        FRAMES_DIR = os.path.join(os.path.abspath(''), "frames", *folders)
        if os.path.exists(FRAMES_DIR):
            rmtree(FRAMES_DIR)
        os.makedirs(FRAMES_DIR)
        the_end = self.total_time() - self.scale
        i = 1
        while self.offset < the_end:
            plt.clf() # Mandatory call, otherwise the figure will not take into account the scale.
            self.fig = self.render_scaled()
            self.fig.savefig(os.path.join(FRAMES_DIR, "{}_{}.png".format(folders[-1], i)))
            self.offset += self.dt_per_sample if not unique else self.scale
            i += 1
            print("{} frames generation: {:.1f}% done.".format("/".join(folders), (self.offset * 100) / the_end))


    def to_csv(self):
        pass
    
    def render(self, color='red'):
        X, Y = self.data()
        ax = sns.lineplot(X, Y, color=color)
        ax.set_title("Realtime System")
        L = ax.lines[-1]
        x1 = L.get_xydata()[:,0]
        y1 = L.get_xydata()[:,1]
        ## Determine ymax from the whole input not truncated by scale/offset
        ## So that the scale doesn't change in the same graph when you change scale/offset
        ymax = max(Y) * 1.05 ## valeur en dur pour qu'il y ait un peu de marge
        ax.set(ylim=(0,ymax))
        ax.fill_between(x1, y1, color=color, alpha=1)
        return ax.figure
    
    def render_scaled(self, color='red'):
        self.data() # Refresh data
        X, Y = self.data_scaled()
        ax = sns.lineplot(X, Y, color=color)
        L = ax.lines[-1]
        x1 = L.get_xydata()[:,0]
        y1 = L.get_xydata()[:,1]
        ## Determine ymax and ymin from the whole input not truncated by scale/offset
        ## So that the scale doesn't change in the same graph when you change scale/offset
        ymax = np.max(Y,axis=None) * 1.05
        ax.set(ylim=(0, ymax))
        ax.fill_between(x1, y1, color=color, alpha=1)
        plt.axis('off')
        fig = ax.figure
        fig.tight_layout(pad=0, h_pad=0, w_pad=0, rect=None)
        return fig

    @staticmethod
    def from_dict(obj: Any) -> 'RealtimeSystem':
        assert isinstance(obj, dict)
        realtime_tick = from_int(obj.get("realtime_tick"))
        dt_per_sample = from_int(obj.get("dt_per_sample"))
        scale = from_int(obj.get("scale"))
        offset = from_int(obj.get("offset"))
        transition_type = TransitionType.from_dict(obj.get("transition_type"))
        noise = Noise.from_dict(obj.get("noise"))
        states = from_list(State.from_dict, obj.get("states"))
        return RealtimeSystem(realtime_tick, dt_per_sample, scale, offset, transition_type, noise, states)

    def to_dict(self) -> dict:
        result: dict = {}
        result["realtime_tick"] = from_int(self.realtime_tick)
        result["dt_per_sample"] = from_int(self.dt_per_sample)
        result["scale"] = from_int(self.scale)
        result["offset"] = from_int(self.offset)
        result["transition_type"] = to_class(
            TransitionType, self.transition_type)
        result["noise"] = to_class(Noise, self.noise)
        result["states"] = from_list(lambda x: to_class(State, x), self.states)
        return result


if __name__ == "__main__":
    realtime_tick = 10 # milliseconds.
    delta_time_per_sample = 20  # seconds
    scale = 2000 # seconds
    offset = 0 # seconds
    transition = TransitionType("sine", 5)
    noise = Noise("gaussian", 0.5)  # Normal function (Gaussian noise)
    # 3 states, with impulses 
    rest, active = states = [State("Rest", 1200, 10, []), State("Active", 1200, 50,[])]

    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          scale,
                          offset,
                          transition,
                          noise,
                          StateFactory.random(10, rest, active, with_random_impulses=True, max_impulses_per_state=1))

    rlts.generate_frames(["train", "stable"])

    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          scale,
                          offset,
                          transition,
                          noise,
                          StateFactory.random(10, rest, active, with_random_impulses=True, random_frequency=0.8, max_impulses_per_state=10))

    rlts.generate_frames(["train", "malfunction"])
    
    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          scale,
                          offset,
                          transition,
                          noise,
                          StateFactory.random(10, rest, active, with_random_impulses=False, max_impulses_per_state=0))

    rlts.generate_frames(["valid","stable"])

    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          scale,
                          offset,
                          transition,
                          noise,
                          StateFactory.random(10, rest, active, with_random_impulses=True, random_frequency=0.8, max_impulses_per_state=10))

    rlts.generate_frames(["valid", "malfunction"])

    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          scale,
                          offset,
                          transition,
                          noise,
                          StateFactory.random(200, rest, active))

    rlts.generate_frames(["object_detection", "stable"], unique=True)
    
    rlts = RealtimeSystem(realtime_tick,
                          delta_time_per_sample,
                          scale,
                          offset,
                          transition,
                          noise,
                          StateFactory.random(200, rest, active, with_random_impulses=True, random_frequency=0.8, max_impulses_per_state=10))

    rlts.generate_frames(["object_detection", "malfunction"], unique=True)
