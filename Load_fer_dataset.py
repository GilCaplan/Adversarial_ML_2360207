# File: data_loader.py
import kagglehub
import os
import shutil

def load_fer_data(local_folder="/Users/USER/Desktop/University/Semester 7/CS Adverserial ML/Project/fer_dataset"):
    # Ensure the path is relative to where the script runs
    dest_path = os.path.join(os.getcwd(), local_folder)
    
    # Download and move if not present locally
    if not (os.path.exists(dest_path) and os.listdir(dest_path)):
        print("Downloading dataset...")
        cached_path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
        shutil.copytree(cached_path, dest_path, dirs_exist_ok=True)
        print("Download and move complete.")
    
    # Define emotions
    base_dir = os.path.join(dest_path, "train")
    emotions = ["happy", "sad", "neutral", "angry", "disgusted", "fearful", "surprised"]
    data = {emo: [] for emo in emotions}

    # Load images
    for emo in emotions:
        folder = os.path.join(base_dir, emo)
        if os.path.exists(folder):
            data[emo] = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
    print(f"Data loaded successfully.")
    return data["happy"], data["sad"], data["neutral"], data["angry"], data["disgusted"], data["fearful"], data["surprised"]