def calculate_average(values):
    """Calculate the average of a list of numbers."""
    if not values:
        return 0
    return sum(values) / len(values)