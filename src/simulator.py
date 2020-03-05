import json
from typing import Any, List, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


class Noise:
    type: str
    amplitude: int

    def __init__(self, type: str, amplitude: int) -> None:
        self.type = type
        self.amplitude = amplitude

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
    pass
