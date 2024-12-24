from gui import HousePricePredictorGUI
from models import train_models
import tkinter as tk

def main():
    # First train/load the models
    print("Training models...")
    train_models()
    print("Model training complete!")
    
    # Create and run the GUI
    print("Launching GUI...")
    root = tk.Tk()
    app = HousePricePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
