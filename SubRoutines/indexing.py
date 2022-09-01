def find_matching_index(full_lat, target_lat):
    return (abs(full_lat - target_lat)).argmin()
