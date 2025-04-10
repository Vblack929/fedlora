# Step 1: Install and set up PyDrive
# pip install pydrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import torch
import os
import shutil

# Authenticate
gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not available
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth)

# Step 2: Create a folder in Google Drive for the model
folder_metadata = {
    'title': 'Base_Model',
    'mimeType': 'application/vnd.google-apps.folder'
}
folder = drive.CreateFile(folder_metadata)
folder.Upload()

print(f"Created folder in Google Drive with ID: {folder['id']}")

# Step 3: Upload each file from the base_model directory
model_dir = "save/base_model"  # Adjust this path if your base_model is elsewhere
for filename in os.listdir(model_dir):
    file_path = os.path.join(model_dir, filename)
    
    # Skip directories
    if os.path.isdir(file_path):
        continue
    
    # Create file in the folder we created
    file_metadata = {
        'title': filename,
        'parents': [{'id': folder['id']}]
    }
    
    file_drive = drive.CreateFile(file_metadata)
    file_drive.SetContentFile(file_path)
    file_drive.Upload()
    
    print(f"Uploaded {filename} to Google Drive")

print(f"Model files uploaded to Google Drive folder: FL_LoRA_Model (ID: {folder['id']})")

# Alternatively, create a zip archive and upload that
"""
# Create a zip archive of the model directory
zip_path = "base_model.zip"
shutil.make_archive("base_model", 'zip', model_dir)

# Upload the zip file
file_drive = drive.CreateFile({'title': 'base_model.zip'})
file_drive.SetContentFile(zip_path)
file_drive.Upload()

print(f"Model archive uploaded to Google Drive with ID: {file_drive['id']}")

# Cleanup the temporary zip file
os.remove(zip_path)
"""
