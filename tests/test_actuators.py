import json
import time

import pytest
import yaml

from src.actuators.client import ActuatorClient
from src.actuators.mock import MockBroker
from src.actuators.registry import Command, check, intent_to_command


def test_valid_commands_pass():
    for room in ["kitchen", "bedroom", "livingroom"]:
        assert check(Command(f"{room}_lights", "on")) is None
        assert check(Command(f"{room}_lights", "off")) is None


def test_bad_commands_rejected():
    assert "unknown device" in check(Command("garage_door", "open"))
    assert "unknown device" in check(Command("washroom_lights", "on"))
    assert "unknown device" in check(Command("bedroom_curtains", "open"))   # out of scope for now
    assert "cannot" in check(Command("kitchen_lights", "brightness", 40))
    assert "no value" in check(Command("kitchen_lights", "on", 5))


def test_mock_receives_exact_payload():
    broker = MockBroker()
    client = ActuatorClient(broker, "home")
    sent, why = client.send(Command("livingroom_lights", "off"))
    assert sent and why == "ok"
    topic, payload = broker.messages[0]
    assert topic == "home/livingroom_lights/set"
    assert json.loads(payload) == {"action": "off", "value": None}
    assert payload == '{"action": "off", "value": null}'


def test_invalid_command_never_reaches_broker():
    broker = MockBroker()
    sent, why = ActuatorClient(broker).send(Command("washroom_lights", "on"))
    assert not sent and "unknown device" in why
    assert broker.messages == []


def test_subscriber_gets_message():
    broker = MockBroker()
    got = []
    broker.subscribe("home/", lambda t, p: got.append((t, p)))
    ActuatorClient(broker).send(Command("bedroom_lights", "on"))
    assert got == [("home/bedroom_lights/set", '{"action": "on", "value": null}')]


def test_intent_mapping():
    c = intent_to_command("activate", "lights", "kitchen")
    assert (c.device, c.action) == ("kitchen_lights", "on")
    c = intent_to_command("deactivate", "lights", "bedroom")
    assert (c.device, c.action) == ("bedroom_lights", "off")


def test_no_room_means_living_room():
    c = intent_to_command("activate", "lights", "none")      # "turn on the lights"
    assert (c.device, c.action) == ("livingroom_lights", "on")


def test_out_of_scope_intents_give_no_command():
    assert intent_to_command("activate", "lights", "washroom") is None   # no such device
    assert intent_to_command("activate", "music", "none") is None
    assert intent_to_command("bring", "newspaper", "none") is None
    assert intent_to_command("increase", "heat", "none", 22) is None     # thermostat is out of scope
    assert intent_to_command("activate", "curtains", "bedroom") is None  # so are curtains


@pytest.mark.device
def test_light_toggles():
    from src.actuators.gpio import GpioActuators
    cfg = yaml.safe_load(open("configs/device.yaml"))["actuators"]["gpio"]
    act = GpioActuators(cfg)
    act.apply(Command("kitchen_lights", "on"))
    assert act.light.value == 1.0
    time.sleep(0.5)
    act.apply(Command("kitchen_lights", "off"))
    assert act.light.value == 0.0
