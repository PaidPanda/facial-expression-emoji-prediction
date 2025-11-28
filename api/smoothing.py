from collections import deque
from config import HISTORY_SIZE

class Smoother:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_SIZE)

    def update(self, label):
        self.history.append(label)
        return max(set(self.history), key=self.history.count)