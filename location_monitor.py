from satellite_data import Satellite

def check_position(satellite):
    """
    Compares the satellite's current location to its target location.
    
    Args:
        satellite (Satellite): The Satellite object to check.
    
    Returns:
        bool: True if the satellite is at the target location, False otherwise.
    """
    # Using a small tolerance for floating-point comparison
    tolerance = 1e-6
    current = satellite.get_location()
    target = satellite._target_location # Accessing the protected attribute for comparison
    
    is_on_target = all(abs(current[i] - target[i]) < tolerance for i in range(3))
    
    return is_on_target