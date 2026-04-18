def calculate_severity(volumes):

    total_volume = volumes["Total Tumor Volume"]

    if total_volume < 50:
        return "Low Tumor Burden (Early Stage)"

    elif total_volume < 120:
        return "Moderate Tumor Burden"

    else:
        return "High Tumor Burden (Critical Stage)"