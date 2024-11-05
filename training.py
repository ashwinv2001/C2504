import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Bidirectional, LSTM, Dense, Dropout, Flatten, BatchNormalization, TimeDistributed, Multiply, Reshape
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
from tensorflow.keras.utils import plot_model

# Parameters
audio_directory = 'GTZAN-sample'  
n_classes = 5
sample_rate = 22050 
max_length = 30 * sample_rate  
bilstm_units = 64 

# Function to extract features from audio files
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))
    
    # Generate the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram.T  

# Load data and extract features
def load_data(audio_directory):
    X, y = [], []
    genres = os.listdir(audio_directory)
    for genre in tqdm(genres, desc="Loading data"):
        genre_folder = os.path.join(audio_directory, genre)
        if os.path.isdir(genre_folder):
            for file_name in os.listdir(genre_folder):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(genre_folder, file_name)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(genre)
    return np.array(X), np.array(y)

# Load and preprocess data
X, y = load_data(audio_directory)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, 'label_encoder.pkl')

y_encoded = tf.keras.utils.to_categorical(y_encoded, num_classes=n_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Reshape data for the Conv2D input
X_train = np.expand_dims(X_train, axis=-1)  
X_test = np.expand_dims(X_test, axis=-1)   

# Attention mechanism
def attention_layer(inputs):
    # Compute attention scores
    attention = Dense(inputs.shape[-2], activation='tanh')(inputs)
    attention = Dense(inputs.shape[-1], activation='softmax')(attention)
    
    # Reshape attention scores to match input shape
    attention = Reshape((inputs.shape[-2], inputs.shape[-1]))(attention)
    
    # Apply attention to inputs
    attention_output = Multiply()([inputs, attention])
    
    return attention_output

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = TimeDistributed(Flatten())(x)
    
    # BiLSTM layers for temporal feature extraction
    x = Bidirectional(LSTM(bilstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Attention layer
    x = attention_layer(x)
    
    x = Bidirectional(LSTM(bilstm_units, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Custom classification layers
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    # Full model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = create_model(input_shape)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', metrics=['accuracy'])
#plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint=tf.keras.callbacks.ModelCheckpoint('best_gtzan_5_custom_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stopping,model_checkpoint])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the model
model.save('model_gtzan_5_custom_with_attention1.h5')
