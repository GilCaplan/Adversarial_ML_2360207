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

def load_weather_data(local_folder_name="weather_dataset"):
    """
    Downloads Weather dataset and returns lists of file paths for each weather type.
    """
    # 1. Define destination path relative to where the script is running
    dest_path = os.path.join(os.getcwd(), local_folder_name)
    
    # 2. Check if dataset exists locally.
    # Unlike the FER dataset, this specific weather dataset typically has classes 
    # directly in the root or inside a 'dataset' subfolder. We check for the 'cloudy' folder.
    
    # We define the specific categories seen in your image
    weather_types = ["cloudy", "rain", "shine", "sunrise"]
    
    # Check if the first category exists to verify the dataset is there
    if not (os.path.exists(os.path.join(dest_path, weather_types[0]))):
        print(f"Dataset not found at {dest_path}. Downloading...")
        try:
            # Downloading the 'Multi-class Weather Dataset' which matches your folders
            cached_path = kagglehub.dataset_download("pratik2901/multiclass-weather-dataset")
            print(f"Copying data to {dest_path}...")
            
            # This specific kaggle dataset often downloads as a folder containing a 'dataset' subfolder.
            # We handle both cases to ensure your local folder looks exactly like your image.
            source_dataset_folder = os.path.join(cached_path, "dataset")
            if os.path.exists(source_dataset_folder):
                shutil.copytree(source_dataset_folder, dest_path, dirs_exist_ok=True)
            else:
                shutil.copytree(cached_path, dest_path, dirs_exist_ok=True)
                
            print("Download and move complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return [], [], [], []
    else:
        print(f"Dataset found at {dest_path}. Skipping download.")

    data = {w_type: [] for w_type in weather_types}

    # 3. Load file paths
    # Note: We look directly in dest_path based on your image structure (no 'train' subfolder)
    for w_type in weather_types:
        folder = os.path.join(dest_path, w_type)
        if os.path.exists(folder):
            data[w_type] = [
                os.path.join(folder, f) 
                for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
    print(f"Data successfully loaded.")
    print(f"Counts: Cloudy={len(data['cloudy'])}, Rain={len(data['rain'])}, Shine={len(data['shine'])}, Sunrise={len(data['sunrise'])}")

    # 4. Return exactly 4 items corresponding to the folders in your image
    return (
        data["cloudy"], 
        data["rain"], 
        data["shine"], 
        data["sunrise"]
    )