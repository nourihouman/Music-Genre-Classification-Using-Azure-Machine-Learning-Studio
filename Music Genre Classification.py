#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
tracks = pd.read_csv('tracks.csv', low_memory=False)



tracks


# In[2]:


fma_small_tracks = tracks[tracks['set.1'] == 'small']
fma_small_tracks


# In[10]:


from azureml.core import Workspace, Dataset, Datastore

subscription_id = '182efa3b-ee4e-4dfa-9c40-6463835d3acb'
resource_group = 'hn250-rg'
workspace_name = 'fma2'

workspace = Workspace(subscription_id, resource_group, workspace_name)

datastore = Datastore.get(workspace, "fmadatastore2")

audio_dataset = Dataset.File.from_files(path=(datastore, 'fma_small/**/*.mp3'))

# Download the audio files to the local directory
audio_files = audio_dataset.download(target_path='.', overwrite=True)


# In[3]:


import os

# Specify your local directory path where the audio files are located
local_directory_path = '.'  # Replace with your actual directory path if different

# Initialize a list to hold all the audio file paths
audio_files = []

# Walk through the directory
for dirpath, dirnames, files in os.walk(local_directory_path):
    # Filter for mp3 files and join the path
    for file in files:
        if file.endswith('.mp3'):
            audio_files.append(os.path.join(dirpath, file))

# Now, 'audio_files' contains the paths to all the mp3 files in the directory
print(f"Found {len(audio_files)} audio files.")


# In[9]:


audio_track_ids = set()
for subdir in os.listdir(local_directory_path):
    subdir_path = os.path.join(local_directory_path, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith('.mp3'):
                track_id = extract_track_id_from_path(os.path.join(subdir_path, file))
                audio_track_ids.add(track_id)




df_track_ids_set = set(fma_small_tracks['track_id'])

# Find extra track IDs in audio files that are not in DataFrame
extra_track_ids = audio_track_ids - df_track_ids_set

# Print extra track IDs
print("Extra Track IDs in Audio Files:", extra_track_ids)

# Check if there are any extra track IDs
if not extra_track_ids:
    print("No extra track IDs found in audio files.")


# In[12]:


import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load an example audio file
y, sr = librosa.load(audio_files[50])

# Create a time array
time = np.arange(0, len(y)) / sr

# Plot the waveform
plt.figure(figsize=(14, 5))
plt.plot(time, y)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()


# In[13]:


y, sr = librosa.load(audio_files[50], sr=22050)

# Generate Mel-spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

# Convert to log scale (dB)
log_S = librosa.power_to_db(S, ref=np.max)

# Plot the Mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.show()


# In[5]:


import os

def extract_track_id_from_path(audio_path):
    """
    Extracts the track ID from the audio file path.
    """
    # Get the base name of the file (e.g., '12345.mp3' or 'track_12345_genre.mp3')
    file_name = os.path.basename(audio_path)

    # Scenario 1: If the entire file name (minus extension) is the track ID
    track_id_os = os.path.splitext(file_name)[0]
    formatted_track_id = track_id_os.lstrip('0')
    
    track_id_int = str(formatted_track_id) if formatted_track_id else 0

    return track_id_int


# In[6]:


fma_small_tracks['track_id'] = fma_small_tracks['track_id'].astype(str)


# In[7]:


import gc
import numpy as np

mel_spectrograms=[]
num_of_samples=1293
for idx, audio_path in enumerate (audio_files):
    try:
        # Extract track ID from the audio_path 
        track_id = extract_track_id_from_path(audio_path)
        if track_id in fma_small_tracks['track_id'].values:
            genre = fma_small_tracks.loc[fma_small_tracks['track_id'] == track_id, 'track.7'].values[0]
        else:
            print(f"Track ID {track_id} not found in the DataFrame.")
            continue
        
        
        y, sr= librosa.load(audio_path, sr=22050)
        duration=librosa.get_duration(y=y, sr=sr)
        
        if duration>=28:
            S=librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,n_mels=128, win_length=512)
            log_S=librosa.power_to_db(S, ref=np.max)
            if log_S.shape[1] != num_of_samples:
                log_S = librosa.util.fix_length(log_S, size=num_of_samples, axis=1, constant_values=(0, -80.0))

            mel_spectrograms.append({'track_id': track_id, 'track.7': genre, 'mel_spectrogram':log_S})
        else:
            print(f"File {audio_path} is shorter than 28 seconds")
        
        del y, S, log_S
        gc.collect()
    except Exception as e:
        print( f'Error Processing file {audio_path}: {e}')
        
    if idx % 100==0:
        print(f"Processed {idx} Files")
        
        batch_number = idx // 100
        np.save(f'mel_spectrograms_batch_{batch_number}.npy', mel_spectrograms)
        mel_spectrograms = [] 


# In[29]:


loaded=np.load('mel_spectrograms_batch_80.npy',allow_pickle=True)
len(loaded)


# In[1]:


import numpy as np
import os

# Directory where .npy files are stored
npy_directory = '.'

# Initialize a list to store all spectrograms
all_spectrograms = []
all_labels=[]

# Iterate over each .npy file in the directory
for file_name in os.listdir(npy_directory):
    if file_name.endswith('.npy'):
        # Load the current file
        file_path = os.path.join(npy_directory, file_name)
        mel_spectrograms = np.load(file_path, allow_pickle=True)

        # Process each spectrogram in the current file
        for d in mel_spectrograms:
            spectrogram = d['mel_spectrogram']
            all_spectrograms.append(spectrogram)
            all_labels.append(d['track.7'])
            
        

            

# Convert the list of padded spectrograms to a NumPy array

mel_spectrograms_array = np.array(all_spectrograms)


# In[2]:


# Normalization of data In Mel spectogram
mel_spectrograms_normalize = (mel_spectrograms_array - np.min(mel_spectrograms_array)) / (np.max(mel_spectrograms_array) - np.min(mel_spectrograms_array))


# In[3]:


#Adding axis to the 2D array
mel_spectrograms_padded= mel_spectrograms_normalize[..., np.newaxis]


# In[4]:


from sklearn.preprocessing import LabelEncoder

genres = np.array(all_labels)

# Encode the genre labels
label_encoder = LabelEncoder()
genres_encoded = label_encoder.fit_transform(genres)
len(genres_encoded)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test_valid, y_train, y_test_valid = train_test_split( mel_spectrograms_padded, genres_encoded, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(X_test_valid,y_test_valid ,test_size=0.5)

print(f'Train: {X_train.shape} {y_train.shape}')
print(f'Valid: {x_val.shape}   {y_val.shape}')
print(f'Test:  {x_test.shape}  {y_test.shape}')


# In[6]:


from tensorflow.keras import models, layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

# Create the CNN model
model = models.Sequential()

# Add the first convolutional layer
model.add(layers.Conv2D(36, (3, 4), input_shape=(128, 1293, 1)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 4)))
model.add(layers.Dropout(0.25))

# Add another convolutional layer
model.add(layers.Conv2D(64, (3, 4)))
model.add(layers.Activation('relu'))          
model.add(layers.MaxPooling2D((2, 4)))
model.add(layers.Dropout(0.25))



# Add a dense layer for classification
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))



model.add(layers.Flatten())

model.add(layers.Dense(len(np.unique(genres_encoded)), activation='softmax'))

optimizer = optimizers.Adam(learning_rate = 0.001)
# Compile the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()


# In[7]:


epochs = 20
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(x=X_train, 
                    y=y_train,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    shuffle=True,callbacks=[early_stopping])


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Set the style of seaborn
sns.set(style='whitegrid')

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r--', label='Training acc')  # Red dashed line for training accuracy
plt.plot(epochs, val_acc, 'b--', label='Validation acc')  # Blue dashed line for validation accuracy
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r--', label='Training loss')  # Red dashed line for training loss
plt.plot(epochs, val_loss, 'b--', label='Validation loss')  # Blue dashed line for validation loss
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

