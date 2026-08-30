"""Nutrient clouds and continuous food regrowth for the living world."""

import math
import random

from food import Food


class NutrientCloud:
    """A drifting nutrient-rich region that spawns food pellets."""

    def __init__(self, x, y, radius, intensity, world_width, world_height):
        # Cloud center in world coordinates.
        self.position = (float(x), float(y))
        # Influence radius for food spawn attempts.
        self.radius = float(radius)
        # Spawn probability multiplier (0..1 typical).
        self.intensity = float(intensity)
        # Arena bounds used for toroidal wrap.
        self.world_width = world_width
        self.world_height = world_height
        # Slow random drift direction.
        self.drift_angle = random.uniform(0, 2 * math.pi)
        self.drift_speed = random.uniform(0.15, 0.45)

    def tick(self):
        """Drift the cloud and wrap at world edges."""
        x, y = self.position
        x += math.cos(self.drift_angle) * self.drift_speed
        y += math.sin(self.drift_angle) * self.drift_speed
        if x < 0:
            x += self.world_width
        elif x >= self.world_width:
            x -= self.world_width
        if y < 0:
            y += self.world_height
        elif y >= self.world_height:
            y -= self.world_height
        self.position = (x, y)
        if random.random() < 0.01:
            self.drift_angle = random.uniform(0, 2 * math.pi)


class FoodEcology:
    """Manage food pellets and nutrient-cloud-driven regrowth."""

    def __init__(self, sim_config):
        # Arena dimensions.
        self.world_width = sim_config["environment_width"]
        self.world_height = sim_config["environment_height"]
        # Target equilibrium food count.
        self.target_density = int(sim_config.get("food_target_density", 75))
        # Base spawn rate per cloud per tick.
        self.base_spawn_rate = float(sim_config.get("nutrient_cloud_spawn_rate", 0.04))
        # Scarcity tuning.
        self.scarcity_threshold = float(sim_config.get("food_scarcity_threshold", 0.35))
        self.scarcity_multiplier = float(
            sim_config.get("food_regrowth_scarcity_multiplier", 2.5)
        )
        # Live food pellets.
        self.food_items = []
        # Nutrient clouds that drive regrowth.
        self.clouds = []
        self._seed_clouds(sim_config)
        self._seed_initial_food(sim_config)

    def _seed_clouds(self, sim_config):
        """Place nutrient clouds across the world."""
        count = int(sim_config.get("nutrient_cloud_count", 10))
        min_r = float(sim_config.get("nutrient_cloud_min_radius", 150))
        max_r = float(sim_config.get("nutrient_cloud_max_radius", 350))
        margin = max_r + 20
        for _ in range(count):
            x = random.uniform(margin, self.world_width - margin)
            y = random.uniform(margin, self.world_height - margin)
            radius = random.uniform(min_r, max_r)
            intensity = random.uniform(0.55, 1.0)
            self.clouds.append(
                NutrientCloud(
                    x, y, radius, intensity, self.world_width, self.world_height
                )
            )

    def _seed_initial_food(self, sim_config):
        """Populate starting food near random clouds and open ground."""
        count = int(sim_config.get("num_food_items", self.target_density))
        for _ in range(count):
            if self.clouds and random.random() < 0.7:
                cloud = random.choice(self.clouds)
                self.food_items.append(self._spawn_near_cloud(cloud))
            else:
                self.food_items.append(self._spawn_random())

    def _spawn_random(self):
        """Spawn one food pellet at a random valid location."""
        margin = 10
        x = random.randint(margin, max(margin + 1, self.world_width - margin))
        y = random.randint(margin, max(margin + 1, self.world_height - margin))
        return Food(x, y)

    def _spawn_near_cloud(self, cloud):
        """Spawn food within a nutrient cloud's radius."""
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, cloud.radius * 0.85)
        x = int(cloud.position[0] + math.cos(angle) * dist)
        y = int(cloud.position[1] + math.sin(angle) * dist)
        x = max(5, min(x, self.world_width - 5))
        y = max(5, min(y, self.world_height - 5))
        return Food(x, y)

    def active_food_count(self):
        """Count food pellets that have not been consumed."""
        return sum(1 for food in self.food_items if food.position is not None)

    def scarcity_ratio(self):
        """Return current food count divided by target density."""
        if self.target_density <= 0:
            return 1.0
        return self.active_food_count() / float(self.target_density)

    def tick(self):
        """Drift clouds and attempt food regrowth from nutrient fields."""
        for cloud in self.clouds:
            cloud.tick()
        ratio = self.scarcity_ratio()
        spawn_boost = 1.0
        if ratio < self.scarcity_threshold:
            spawn_boost = self.scarcity_multiplier
        cap = int(self.target_density * 1.6)
        if self.active_food_count() >= cap:
            return
        for cloud in self.clouds:
            rate = self.base_spawn_rate * cloud.intensity * spawn_boost
            if random.random() < rate:
                self.food_items.append(self._spawn_near_cloud(cloud))
                if self.active_food_count() >= cap:
                    break

    def prune_consumed(self):
        """Drop consumed food entries to keep the list bounded."""
        if len(self.food_items) > self.target_density * 3:
            self.food_items = [
                food for food in self.food_items if food.position is not None
            ]
