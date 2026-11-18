"""Drive the pi hardware directly: the LED strip (pwm output through the mosfet).

Only runs on the pi. gpiozero is imported inside so a laptop can still import the repo.
Scope is lights on and off. With one LED strip wired, every room's light command drives the same LED (demo setup).
"""
import json

from src.actuators.registry import Command, check


class GpioActuators:
    def __init__(self, gpio_cfg: dict):
        from gpiozero import PWMLED
        self.light = PWMLED(gpio_cfg["led_pin"])

    def apply(self, cmd: Command) -> None:
        err = check(cmd)
        if err:
            raise ValueError(err)
        if cmd.action == "on":
            self.light.value = 1.0
        elif cmd.action == "off":
            self.light.value = 0.0


def serve_mqtt(gpio_cfg: dict, mqtt_cfg: dict) -> None:
    """Listen on home/+/set and apply commands. Run this on the pi next to the broker."""
    import paho.mqtt.client as mqtt
    act = GpioActuators(gpio_cfg)

    def on_message(client, userdata, msg):
        device = msg.topic.split("/")[1]
        body = json.loads(msg.payload)
        try:
            act.apply(Command(device, body["action"], body["value"]))
        except ValueError as e:
            print("rejected:", e)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(mqtt_cfg["host"], mqtt_cfg["port"])
    client.subscribe(f"{mqtt_cfg['topic_prefix']}/+/set")
    client.loop_forever()
