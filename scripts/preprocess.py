import csv

# Function to process each line and extract the required fields
def process_line(line):
    parts = line.split()
    timestamp = parts[1]
    can_id = f"{parts[3]} {parts[4]}"
    dlc = parts[6]
    data = parts[7:15] + [""] * (8 - len(parts[7:15]))  # Ensure there are always 8 data fields
    return [timestamp, can_id, dlc] + data

# Read the input text file and process the lines
input_file = 'data/Attack_free_dataset.txt'
output_file = 'data/source_dataset.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(['Timestamp', 'CAN ID', 'DLC', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]'])
    
    for line in infile:
        csv_writer.writerow(process_line(line))

print(f"Data has been written to {output_file}")
