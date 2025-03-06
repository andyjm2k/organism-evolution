import random

class Food:
    def __init__(self, x, y, log_creation=False):
        self.position = (x, y)
        if log_creation:
            print(f"Food placed at position {self.position}.")
