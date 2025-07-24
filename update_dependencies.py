import subprocess
import sys

def update_dependencies():
    """
    Update transformers and other required packages to their latest versions.
    """
    print("Updating transformers and dependencies...")
    packages = ["transformers", "torch", "accelerate"]
    
    for package in packages:
        print(f"Updating {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
    
    print("All dependencies updated successfully.")

if __name__ == "__main__":
    update_dependencies() 