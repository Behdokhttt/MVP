import pickle

# Load the pickle file
with open('data/output/transcript_features/v_001_transcript_features.pkl', 'rb') as f:
    data = pickle.load(f)

# Print information about the data
print(f"Type of data: {type(data)}")

if isinstance(data, dict):
    print(f"Dictionary keys: {list(data.keys())}")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape = {value.shape}")
            if hasattr(value, 'dtypes'):
                print(f"    dtypes: {value.dtypes}")
            elif hasattr(value, 'dtype'):
                print(f"    dtype = {value.dtype}")
        else:
            print(f"  {key}: type = {type(value)}, size = {len(value) if hasattr(value, '__len__') else 'N/A'}")
elif hasattr(data, 'shape'):
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
else:
    print(f"Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    print(f"Type: {type(data)}")
