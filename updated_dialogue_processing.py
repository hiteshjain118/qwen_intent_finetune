import json
import os
import pandas as pd

folder_path = "dstc8-schema-guided-dialogue/train"
rows_with_frames = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and 'dialogues_' in file_path:
        with open(file_path) as f:
            dialogues = json.load(f)
        for dialog in dialogues:
            dialogue_id = dialog["dialogue_id"]
            services = dialog.get("services", [])
            for turn_idx, turn in enumerate(dialog["turns"]):
                # Extract frames information
                frames = turn.get("frames", [])
                frame_info = []
                for frame in frames:
                    frame_data = {
                        "service": frame.get("service", ""),
                        "actions": frame.get("actions", []),
                        "slots": frame.get("slots", []),
                        "state": frame.get("state", {})
                    }
                    frame_info.append(frame_data)
                
                rows_with_frames.append({
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_idx,
                    "speaker": turn["speaker"],
                    "utterance": turn["utterance"],
                    "services": services,
                    "frames": frame_info
                })

df_with_frames = pd.DataFrame(rows_with_frames)
print("DataFrame with frames:")
print(df_with_frames.head())
print(f"\nTotal rows: {len(df_with_frames)}")
print(f"Columns: {list(df_with_frames.columns)}")

# Show an example of frames data
print("\nExample frames data from first row:")
if len(df_with_frames) > 0 and len(df_with_frames.iloc[0]['frames']) > 0:
    print(json.dumps(df_with_frames.iloc[0]['frames'], indent=2))

# Function to serialize dialogue with frames information
def serialize_dialogue_with_frames(df):
    ret = ""
    for _, row in df.iterrows():
        ret += (f"{row['speaker']} : {row['utterance']}\n")
        if row['frames']:
            ret += "  Frames:\n"
            for frame in row['frames']:
                ret += f"    Service: {frame['service']}\n"
                if frame['actions']:
                    ret += f"    Actions: {frame['actions']}\n"
                if frame['slots']:
                    ret += f"    Slots: {frame['slots']}\n"
                if frame['state']:
                    ret += f"    State: {frame['state']}\n"
        ret += "\n"
    return ret

# Example usage with hotel dialogues
hotel_rows_with_frames = df_with_frames[df_with_frames["services"].apply(lambda x: "Hotels_1" in x or "Hotels_2" in x)]
multi_service_with_frames = hotel_rows_with_frames[hotel_rows_with_frames["services"].apply(lambda x: len(x) > 1)]

if len(multi_service_with_frames) > 0:
    first_multi_service_dialog = multi_service_with_frames['dialogue_id'].iloc[0]
    example_multi_service_with_frames = multi_service_with_frames[multi_service_with_frames['dialogue_id'].apply(lambda x: x == first_multi_service_dialog)]
    
    print(f"\nMulti-service hotel dialogues: {len(multi_service_with_frames['dialogue_id'].unique())}")
    print("Example dialogue with frames:")
    print(serialize_dialogue_with_frames(example_multi_service_with_frames)) 