import pandas as pd

def preprocess_can_data(file_path):
    # Read the raw CAN data from the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to hold the data
    timestamps = []
    can_ids = []
    dlcs = []
    data_bytes = []

    # Process each line in the file
    for line in lines:
        parts = line.split()
        if len(parts) < 5:  # Skip lines that don't have enough parts
            continue

        timestamp = float(parts[1])
        can_id = int(parts[3], 16)  # Convert HEX ID to decimal
        dlc = int(parts[5])

        # Extract data bytes and pad with zeros if necessary
        data = [int(byte, 16) for byte in parts[7:7+dlc]]
        data.extend([0] * (8 - len(data)))  # Pad with zeros if DLC < 8

        # Append to lists
        timestamps.append(timestamp)
        can_ids.append(can_id)
        dlcs.append(dlc)
        data_bytes.append(data)

    # Create a DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'CAN_ID': can_ids,
        'DLC': dlcs,
        'DATA_0': [data[0] for data in data_bytes],
        'DATA_1': [data[1] for data in data_bytes],
        'DATA_2': [data[2] for data in data_bytes],
        'DATA_3': [data[3] for data in data_bytes],
        'DATA_4': [data[4] for data in data_bytes],
        'DATA_5': [data[5] for data in data_bytes],
        'DATA_6': [data[6] for data in data_bytes],
        'DATA_7': [data[7] for data in data_bytes],
    })

    # Save preprocessed data
    df.to_csv('../data/preprocessed_can_data.csv', index=False)
    print("Data preprocessing complete. Preprocessed data saved to ../data/preprocessed_can_data.csv")

def load_real_can_data(preprocessed_file_path):
    return pd.read_csv(preprocessed_file_path).values

if __name__ == "__main__":
    preprocess_can_data('../data/can_data.txt')
