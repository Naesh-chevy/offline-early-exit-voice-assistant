"""Energy logging interface.

If there is no INA226 (or we are not on a pi) we return "not measured". Never a made up number.
On android the plan is BatteryManager readings, not wired yet so it also reports "not measured".
"""
import threading
import time

NOT_MEASURED = "not measured"


class NoEnergy:
    """Stub used when no meter is available."""
    name = "none"

    def start(self) -> None:
        pass

    def stop(self) -> dict:
        return {"energy_mj": NOT_MEASURED, "avg_power_mw": NOT_MEASURED, "energy_source": self.name}


class INA226:
    """Reads the INA226 over i2c in a thread and integrates power. Needs smbus2 and a real chip."""
    name = "ina226"

    def __init__(self, bus: int = 1, addr: int = 0x40, shunt_ohms: float = 0.1, period_s: float = 0.002):
        import smbus2  # only exists on the device
        self.bus = smbus2.SMBus(bus)
        self.addr, self.shunt, self.period = addr, shunt_ohms, period_s
        self.samples = []
        self.t0 = 0.0
        self._stop = threading.Event()

    def _word(self, reg: int) -> int:
        raw = self.bus.read_word_data(self.addr, reg)
        return ((raw & 0xFF) << 8) | (raw >> 8)  # chip is big endian

    def read_power_mw(self) -> float:
        shunt_v = self._word(0x01)
        if shunt_v > 32767:
            shunt_v -= 65536
        bus_v = self._word(0x02) * 1.25e-3          # 1.25 mV per lsb
        amps = shunt_v * 2.5e-6 / self.shunt        # 2.5 uV per lsb
        return bus_v * amps * 1000

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.read_power_mw()))
            time.sleep(self.period)

    def start(self) -> None:
        self.samples, self._stop = [], threading.Event()
        self.t0 = time.perf_counter()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self) -> dict:
        self._stop.set()
        self.th.join()
        dur = time.perf_counter() - self.t0
        if len(self.samples) < 2:
            return {"energy_mj": NOT_MEASURED, "avg_power_mw": NOT_MEASURED, "energy_source": self.name}
        avg = sum(p for _, p in self.samples) / len(self.samples)
        return {"energy_mj": avg * dur, "avg_power_mw": avg, "energy_source": self.name}  # mW * s = mJ


class GpioMarker:
    """Context manager: pin goes high during inference so the separate logger can cut the trace."""

    def __init__(self, pin: int | None):
        self.dev = None
        if pin is not None:
            try:
                from gpiozero import DigitalOutputDevice
                self.dev = DigitalOutputDevice(pin)
            except Exception:  # no gpio here (laptop), marker is just skipped
                self.dev = None

    def __enter__(self):
        if self.dev:
            self.dev.on()
        return self

    def __exit__(self, *exc):
        if self.dev:
            self.dev.off()


def get_energy_logger(device_cfg: dict):
    """INA226 if the config turns it on and it works, else the stub."""
    if device_cfg["target"] != "android" and device_cfg["energy"]["ina226_enabled"]:
        try:
            g = device_cfg["actuators"]["gpio"]
            e = device_cfg["energy"]
            return INA226(g["i2c_bus"], e["ina226_addr"], e["shunt_ohms"])
        except Exception as err:  # smbus2 missing or chip not wired
            print("ina226 not available:", err)
    return NoEnergy()
