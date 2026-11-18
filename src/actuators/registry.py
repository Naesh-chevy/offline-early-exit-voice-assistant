"""Device registry: which devices exist, what actions they allow, which values are legal.

Every command goes through check() before it is sent. This is the safety net after the policy.
Scope for now: lights on and off in the kitchen, bedroom and living room.
"""
from dataclasses import dataclass

# action -> None (no value) or (lo, hi) range the value must be in
DEVICES = {
    "kitchen_lights": {"kind": "lights", "actions": {"on": None, "off": None}},
    "bedroom_lights": {"kind": "lights", "actions": {"on": None, "off": None}},
    "livingroom_lights": {"kind": "lights", "actions": {"on": None, "off": None}},
}

# fsc style intent -> registry action. only the pairs we support are listed
ACTION_MAP = {
    ("lights", "activate"): "on",
    ("lights", "deactivate"): "off",
}

# fsc has no "living room", and "turn on the lights" with no room is labelled location "none".
# so no room said = living room (the default room). washroom has no device, so the registry rejects it.
DEFAULT_ROOM = "livingroom"


@dataclass
class Command:
    device: str
    action: str
    value: float | None = None


def check(cmd: Command) -> str | None:
    """None if the command is fine, otherwise a short reason string."""
    dev = DEVICES.get(cmd.device)
    if dev is None:
        return f"unknown device {cmd.device}"
    if cmd.action not in dev["actions"]:
        return f"{cmd.device} cannot {cmd.action}"
    rng = dev["actions"][cmd.action]
    if rng is None:
        return None if cmd.value is None else f"{cmd.action} takes no value"
    if cmd.value is None:
        return f"{cmd.action} needs a value"
    if not rng[0] <= cmd.value <= rng[1]:
        return f"{cmd.value} outside {rng[0]}..{rng[1]}"
    return None


def intent_to_command(action: str, obj: str, location: str, value: float | None = None) -> Command | None:
    """Map a predicted (action, object, location) to a command. None if there is no such device."""
    reg_action = ACTION_MAP.get((obj, action))
    if reg_action is None:
        return None
    room = DEFAULT_ROOM if location == "none" else location
    device = f"{room}_{obj}"
    if device not in DEVICES:
        return None
    return Command(device, reg_action)
