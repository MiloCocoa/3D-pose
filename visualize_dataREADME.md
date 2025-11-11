# How to Visualize Your 3D Pose Data

I've provided two scripts to help you visualize the data from your `.pickle` file:

1. `create_mock_data.py`: This script creates a dummy `data/data_3D.pickle` file in a new `data/` directory. It's based on the structure described in your `README.md` and `data_loader.py`.
    
2. `visualize_data.py`: This script loads the `.pickle` file, displays an interactive menu, and renders your chosen sequence in a **fully interactive 3D plot**.
    

## New Features:

- **No `ffmpeg` Needed:** The script no longer saves `.mp4` files.
    
- **3D Orbit Control:** An interactive window will open. **You can click and drag the 3D plot** to rotate your view and inspect the skeleton from any angle.
    
- **Frame Slider:** Use the **slider at the bottom of the window** to scrub through the animation frame by frame.
    
- **Feet Anchored:** The skeleton is now anchored to the "floor" at the midpoint of the feet for a much more stable and realistic visualization.
    

## Steps to Run:

1. Install Dependencies:
    
    Make sure you have matplotlib, pandas, and numpy installed:
    
    ```
    pip install matplotlib pandas numpy
    ```
    
2. Generate Mock Data (Optional, but recommended first):
    
    Run this script to create the data/data_3D.pickle file.
    
    ```
    python create_mock_data.py
    ```
    
    _(You can skip this if you already have your real `data/data_3D.pickle` file in place)._
    
3. Run the Visualization:
    
    This will read the .pickle file and show you a menu of all available sequences.
    
    ```
    python visualize_data.py
    ```
    
    **Example Output:**
    
    ```
    --- Available Sequences to Visualize ---
      [1] squat - Sub: S0 - Rep: 1 - Label: Correct (100 frames)
      [2] squat - Sub: S0 - Rep: 2 - Label: Correct (100 frames)
      ...
    ------------------------------------------
    Enter the number of the sequence to visualize (1-40): 1
    ```
    
4. Interact with the Plot:
    
    An interactive window will pop up.
    
    - **Click and drag** on the skeleton to rotate the view.
        
    - **Use the slider** at the bottom to change frames.
        

This should be a much more powerful tool for you to inspect your data. Let me know if it works!