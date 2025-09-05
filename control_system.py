from satellite_data import Satellite, Thruster

def calculate_and_apply_correction(satellite, thruster, new_current_location):
    """
    Calculates the correction vector and applies it to the satellite's location.
    This simulates the manual input and subsequent correction.

    Args:
        satellite (Satellite): The Satellite object to be corrected.
        thruster (Thruster): The Thruster object to apply the correction.
        new_current_location (list): The new location reported by the operator.
    
    Returns:
        list: The correction vector that was applied.
    """
    # Calculate the correction vector
    target = satellite._target_location
    correction_vector = [target[i] - new_current_location[i] for i in range(3)]

    # Apply the correction using the thruster
    print(f"Correction vector calculated: {correction_vector}")
    
    # Note: The pseudocode suggests this:
    # my_satellite.set_location(new_current_location + correction_vector)
    # The `Thruster.fire` method now encapsulates this logic for better OOP.
    thruster.fire(satellite, correction_vector)
    
    return correction_vector