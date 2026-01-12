import kagglehub
import os
import shutil

def load_fer_data(local_folder_name="fer_dataset"):
    """
    Downloads FER dataset and returns lists of file paths for each emotion.
    Returns paths (strings) to save RAM. The main experiment script handles opening them.
    """
    # 1. Define destination path relative to where the script is running
    dest_path = os.path.join(os.getcwd(), local_folder_name)
    
    # 2. Check if dataset exists locally (specifically looking for the 'train' folder)
    # The dataset structure usually contains a 'train' and 'test' folder.
    train_dir = os.path.join(dest_path, "train")
    
    if not (os.path.exists(train_dir) and os.listdir(train_dir)):
        print(f"Dataset not found at {dest_path}. Downloading...")
        try:
            cached_path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
            print(f"Copying data to {dest_path}...")
            shutil.copytree(cached_path, dest_path, dirs_exist_ok=True)
            print("Download and move complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return [], [], [], [], [], [], []
    else:
        print(f"Dataset found at {dest_path}. Skipping download.")
    
    # 3. Define emotions (matching the specific folder names of this Kaggle dataset)
    emotions = ["happy", "sad", "neutral", "angry", "disgusted", "fearful", "surprised"]
    data = {emo: [] for emo in emotions}

    # 4. Load file paths
    for emo in emotions:
        folder = os.path.join(train_dir, emo)
        if os.path.exists(folder):
            # List comprehension to get full paths of images
            data[emo] = [
                os.path.join(folder, f) 
                for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
    print(f"Data successfully loaded.")
    print(f"Counts: Happy={len(data['happy'])}, Sad={len(data['sad'])}, Neutral={len(data['neutral'])}")

    # Return exactly 7 items to match the unpacking in VLM_experiments.py
    return (
        data["happy"], 
        data["sad"], 
        data["neutral"], 
        data["angry"], 
        data["disgusted"], 
        data["fearful"], 
        data["surprised"]
    )