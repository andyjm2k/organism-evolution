class Food:
    """One pellet. A strain, when present, decides whether the bite helps or hurts."""

    def __init__(self, x, y, strain=None, log_creation=False):
        # None means the pellet has been eaten or uprooted.
        self.position = (x, y)
        # Plant strategy. None is an ordinary indifferent meal.
        self.strain = strain
        # Steps until a grazed symbiotic pellet can be bitten again.
        self.regrow_in = 0
        # Filled by the ecology so a bite can ask for a seed.
        self.ecology = None
        self.cloud = None
        if log_creation:
            print(f"Food placed at position {self.position}.")
