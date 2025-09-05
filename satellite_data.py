import random

class Satellite:
    """
    Represents a satellite with a target and current location.
    The satellite's location is represented as a tuple of (x, y, z) coordinates.
    """
    def __init__(self, target_location):
        """
        Initializes the Satellite object.
        
        Args:
            target_location (tuple): The desired coordinates of the satellite.
        """
        self._target_location = target_location
        self._current_location = list(target_location) # Use a list for mutability
        print(f"Satellite initialized at target location: {self._current_location}")

    def get_location(self):
        """
        Returns the satellite's current location.
        
        Returns:
            list: The current coordinates of the satellite.
        """
        return self._current_location

    def set_location(self, new_coordinates):
        """
        Updates the satellite's current location.
        
        Args:
            new_coordinates (list): The new coordinates for the satellite.
        """
        self._current_location = new_coordinates

    def simulate_drift(self):
        """
        Simulates gravitational drift by adding a small random vector to the
        satellite's current location.
        """
        drift_vector = [random.uniform(-0.1, 0.1) for _ in range(3)]
        self._current_location = [
            self._current_location[i] + drift_vector[i] for i in range(3)
        ]

class Thruster:
    """
    Represents the satellite's propulsion system.
    """
    def __init__(self):
        """
        Initializes the Thruster object.
        """
        print("Thruster system is ready.")
        
    def fire(self, satellite, correction_vector):
        """
        Applies a correction vector to the satellite's location.
        
        Args:
            satellite (Satellite): The Satellite object to be moved.
            correction_vector (list): The vector representing the thrust needed.
        
        Returns:
            list: The satellite's new location after correction.
        """
        new_location = [
            satellite.get_location()[i] + correction_vector[i] for i in range(3)
        ]
        satellite.set_location(new_location)
        return new_location