"""Sends validated commands. Transport is mqtt (or ble, not done yet) from device.yaml."""
import json

from src.actuators.registry import Command, check


class MqttTransport:
    def __init__(self, host: str, port: int):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(host, port)
        self.client.loop_start()

    def publish(self, topic: str, payload: str) -> None:
        self.client.publish(topic, payload, qos=1).wait_for_publish(timeout=2)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


class BleTransport:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ble transport is not written yet, use mqtt")


class ActuatorClient:
    def __init__(self, transport, topic_prefix: str = "home"):
        self.transport = transport
        self.prefix = topic_prefix

    def send(self, cmd: Command) -> tuple[bool, str]:
        """Validate then publish. Returns (sent, reason). Bad commands never reach the broker."""
        err = check(cmd)
        if err:
            return False, err
        payload = json.dumps({"action": cmd.action, "value": cmd.value})
        self.transport.publish(f"{self.prefix}/{cmd.device}/set", payload)
        return True, "ok"


def from_config(cfg: dict) -> ActuatorClient:
    a = cfg["actuators"]
    if a["transport"] == "mqtt":
        t = MqttTransport(a["mqtt"]["host"], a["mqtt"]["port"])
        return ActuatorClient(t, a["mqtt"]["topic_prefix"])
    return ActuatorClient(BleTransport(), "home")
