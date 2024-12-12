import re
import pandas as pd

# Function to process the text and extract data
def process_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Regular expressions to match the different parts of the text
    experiment_pattern = r'(\d+) datapoints, experiment (\d+):'
    epoch_pattern = r'Epoch (\d+): test loss = ([\d.e-]+), lr = ([\d.e-]+)'
    
    # Initialize list to hold structured data
    data = []
    
    # Find all experiment blocks
    experiment_blocks = re.split(r'---+\s*', text.strip())

    for block in experiment_blocks:
        # Find experiment metadata
        experiment_match = re.search(experiment_pattern, block)
        if experiment_match:
            datapoints = int(experiment_match.group(1))
            experiment_number = int(experiment_match.group(2))
            
            # Find all epoch data within the experiment block
            epochs = re.findall(epoch_pattern, block)
            
            for epoch in epochs:
                epoch_number = int(epoch[0])
                test_loss = float(epoch[1])
                learning_rate = float(epoch[2])
                
                # Append data to the list
                data.append({
                    'datapoints': datapoints,
                    'experiment': experiment_number,
                    'epoch': epoch_number,
                    'test_loss': test_loss,
                    'learning_rate': learning_rate
                })
    
    # Convert list to DataFrame
    df = pd.DataFrame(data)
    return df