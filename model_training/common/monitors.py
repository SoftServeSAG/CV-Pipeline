from datetime import datetime


class TrainingMonitor:
    available_methods = ["time", "epochs"]

    def __init__(self, monitor_method: str = "time", interval: int = 600):
        assert monitor_method in self.available_methods

        self.method = monitor_method
        self.counter = 0
        self.interval = interval

    def reset(self):
        if self.method == "time":
            self.counter = datetime.now()
        elif self.method == "epochs":
            self.counter = 0

    def update(self, add_value=1):
        if self.method == "epochs":
            self.counter += add_value

    def should_save_checkpoint(self) -> bool:
        if self.method == "time":
            return (datetime.now() - self.counter).total_seconds() > self.interval
        elif self.method == "epochs":
            return self.counter >= self.interval
