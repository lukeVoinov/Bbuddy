import numpy as np
import pandas as pd
import librosa
from scipy.stats import median_abs_deviation
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import os

def extract_pitch(audio_file):
    """Extract pitch from audio file using librosa's pyin algorithm."""
    try:
        # Load the audio file
        scale, sr = librosa.load(audio_file, sr=None, mono=True)
        
        # Extract pitch using pyin
        pitches, voiced_flag, _ = librosa.pyin(
            scale,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr,
            frame_length=2048,
            hop_length=512
        )
        
        return pitches, voiced_flag
    except Exception as e:
        raise RuntimeError(f"Error extracting pitch: {str(e)}")

def calculate_pitch_statistics(pitches, voiced_flag=None):
    """Calculate pitch statistics using only voiced segments."""
    try:
        # Filter for voiced segments
        if voiced_flag is not None:
            valid_pitches = pitches[voiced_flag]
        else:
            valid_pitches = pitches[pitches > 0]
        
        if len(valid_pitches) == 0:
            return 0, 0
        
        # Calculate median and MAD
        pitch_median = np.median(valid_pitches)
        pitch_mad = median_abs_deviation(valid_pitches, scale='normal')
        
        return pitch_median, pitch_mad
    except Exception as e:
        raise RuntimeError(f"Error calculating pitch statistics: {str(e)}")

def classify_audio(audio_file, model_path):
    """
    Classify an audio file using the trained model.
    
    Parameters:
    audio_file (str): Path to the audio file
    model_path (str): Path to the saved model file
    
    Returns:
    dict: Classification results including features and prediction
    """
    try:
        # Extract pitch features
        pitches, voiced_flag = extract_pitch(audio_file)
        pitch_median, pitch_mad = calculate_pitch_statistics(pitches, voiced_flag)
        
        # Load the trained model
        model = joblib.load(model_path)
        
        # Prepare features for classification
        features = np.array([[pitch_mad, pitch_median]])  # Note the order matches training data
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        # Create result dictionary
        result = {
            'filename': os.path.basename(audio_file),
            'median_pitch': float(pitch_median),
            'pitch_mad': float(pitch_mad),
            'classification': 'parent' if prediction == 1 else 'child',
            'confidence': float(max(probabilities)),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"Classification error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    audio_file = "C:/Users/lukev/Projects/BehavBuddy/audio/leslie_voice.mp3"
    model_path = "C:/Users/lukev/Projects/BehavBuddy/trained_models/trainedBB_Afive.pkl"
    
    try:
        result = classify_audio(audio_file, model_path)
        print("\nClassification Results:")
        print(f"File: {result['filename']}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Median Pitch: {result['median_pitch']:.2f}")
        print(f"Pitch MAD: {result['pitch_mad']:.2f}")
        print(f"Timestamp: {result['timestamp']}")
    
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")