import numpy as np
import yaml

from src.actuators.client import ActuatorClient
from src.actuators.mock import MockBroker
from src.runtime.pipeline import Pipeline

CFG = yaml.safe_load(open("configs/train_encoder.yaml"))
LABELS = {"action": {"activate": 0, "deactivate": 1, "increase": 2},
          "object": {"heat": 0, "lights": 1, "none": 2},
          "location": {"kitchen": 0, "none": 1, "washroom": 2}}


class FakeEncoder:
    """Stands in for SegmentedEncoder, always exits at exit 1 with the given labels."""

    def __init__(self, action, obj, loc, slot=6, p_act=0.99):
        self.out = {"action": np.eye(3)[action] * 9, "obj": np.eye(3)[obj] * 9, "location": np.eye(3)[loc] * 9,
                    "slot": np.eye(15)[slot] * 9, "decision": np.log(np.array([p_act, 0.005, 0.005]))}

    def start(self, mel):
        pass

    def exit_fn(self, k):
        return self.out


def make(enc):
    broker = MockBroker()
    return Pipeline(None, enc, LABELS, ActuatorClient(broker), CFG, CFG["audio"]), broker


def test_confident_light_command_reaches_actuator():
    pipe, broker = make(FakeEncoder(0, 1, 0))     # activate lights kitchen
    info = pipe.handle_utterance(np.zeros(64000, dtype=np.float32))
    assert info["decision"] == "act" and info["sent"] and info["exit"] == 1
    assert broker.messages == [("home/kitchen_lights/set", '{"action": "on", "value": null}')]


def test_no_room_said_goes_to_living_room():
    pipe, broker = make(FakeEncoder(1, 1, 1))     # deactivate lights, location none
    pipe.handle_utterance(np.zeros(64000, dtype=np.float32))
    assert broker.messages == [("home/livingroom_lights/set", '{"action": "off", "value": null}')]


def test_out_of_scope_command_sends_nothing():
    pipe, broker = make(FakeEncoder(2, 0, 1))     # increase heat, thermostat is out of scope
    info = pipe.handle_utterance(np.zeros(64000, dtype=np.float32))
    assert not info["sent"] and broker.messages == []


def test_reject_sends_nothing():
    pipe, broker = make(FakeEncoder(0, 1, 0, p_act=0.0))
    info = pipe.handle_utterance(np.zeros(64000, dtype=np.float32))
    assert info["decision"] == "reject" and not info["sent"]
    assert broker.messages == []


def test_unknown_device_is_blocked_by_registry():
    pipe, broker = make(FakeEncoder(0, 1, 2))     # lights in the washroom, no such device
    info = pipe.handle_utterance(np.zeros(64000, dtype=np.float32))
    assert not info["sent"] and broker.messages == []
