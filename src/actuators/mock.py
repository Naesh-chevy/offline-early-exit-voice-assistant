"""In memory broker for tests. Same publish() as the real mqtt transport."""


class MockBroker:
    def __init__(self):
        self.messages = []   # (topic, payload str)
        self.subs = []       # (prefix, callback)

    def publish(self, topic: str, payload: str) -> None:
        self.messages.append((topic, payload))
        for prefix, cb in self.subs:
            if topic.startswith(prefix):
                cb(topic, payload)

    def subscribe(self, prefix: str, callback) -> None:
        self.subs.append((prefix, callback))
