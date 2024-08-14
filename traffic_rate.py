# Function to calculate traffic rate
def calculate_traffic_rate(frame_count, start_time, end_time):
    duration = end_time - start_time
    fps = frame_count / duration
    traffic_rate = fps * 60  # Assuming 1 frame represents 1 minute of traffic
    return traffic_rate
