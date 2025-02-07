import numpy as np
import pandas as pd
import librosa
from scipy.stats import median_abs_deviation
from sklearn.pipeline import Pipeline
import joblib
import os
import tempfile
import soundfile as sf

def extract_segments_from_audio(audio_path, segments):
    """
    Extract audio segments based on timestamp intervals using librosa.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    try:
        # Load the full audio file
        y, sr = librosa.load(audio_path, sr=None)
        temp_segments = []
        
        for start_time, end_time in segments:
            # Convert times to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract segment
            segment = y[start_sample:end_sample]
            
            # Create temporary file for segment
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, segment, sr)
            temp_segments.append(temp_file.name)
            
        return temp_segments
    except Exception as e:
        raise RuntimeError(f"Error extracting audio segments: {str(e)}")

def parse_transcript(transcript_text):
    """
    Parse transcript text to extract speaker labels and time intervals.
    """
    segments = []
    for line in transcript_text.split('\n'):
        if line.strip() and (':' in line) and ('-' in line):
            try:
                # Split line into speaker and timing
                speaker, timing = line.split(':', 1)
                # Handle the case where there are additional fields after seconds
                time_part = timing.split(';')[0]  # Split on semicolon and take first part
                start_time, end_time = time_part.split('-')
                
                # Convert times to float, handling "seconds" text
                start_time = float(start_time.strip())
                end_time = float(end_time.split('seconds')[0].strip())
                
                segments.append({
                    'speaker': speaker.strip(),
                    'start_time': start_time,
                    'end_time': end_time
                })
            except Exception as e:
                print(f"Warning: Could not parse line: {line}")
                continue
    
    return segments

def extract_pitch(audio_file):
    """Extract pitch from audio file using librosa's pyin algorithm."""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract pitch using pyin
        pitches, voiced_flag, _ = librosa.pyin(
            y,
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
    """Classify an audio file using the trained model."""
    try:
        # Extract pitch features
        pitches, voiced_flag = extract_pitch(audio_file)
        pitch_median, pitch_mad = calculate_pitch_statistics(pitches, voiced_flag)
        
        # Load the trained model
        model = joblib.load(model_path)
        
        # Prepare features for classification
        features = np.array([[pitch_mad, pitch_median]])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        # Create result dictionary
        result = {
            'filename': os.path.basename(audio_file),
            'median_pitch': float(pitch_median),
            'pitch_mad': float(pitch_mad),
            'classification': 'parent' if prediction == 1 else 'child',
            'confidence': float(max(probabilities))
        }
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"Classification error: {str(e)}")

def classify_segments(audio_path, transcript_text, model_path):
    """Classify each segment from the transcript using the trained model."""
    try:
        # Verify files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        # Parse transcript
        segments = parse_transcript(transcript_text)
        
        if not segments:
            raise ValueError("No valid segments found in transcript")
            
        # Extract time intervals
        time_intervals = [(seg['start_time'], seg['end_time']) for seg in segments]
        
        # Extract audio segments
        audio_segments = extract_segments_from_audio(audio_path, time_intervals)
        
        results = []
        
        # Classify each segment
        for segment_file, segment_info in zip(audio_segments, segments):
            try:
                # Classify the segment
                classification = classify_audio(segment_file, model_path)
                
                # Add original speaker and timing information
                classification.update({
                    'original_speaker': segment_info['speaker'],
                    'start_time': segment_info['start_time'],
                    'end_time': segment_info['end_time']
                })
                
                results.append(classification)
                
            except Exception as e:
                print(f"Warning: Could not classify segment {segment_file}: {str(e)}")
                continue
            finally:
                # Clean up temporary file
                try:
                    os.remove(segment_file)
                except:
                    pass
                
        return results
    
    except Exception as e:
        raise RuntimeError(f"Error classifying segments: {str(e)}")

def read_transcript_file(transcript_path):
    """Read and parse a transcript file."""
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found at: {transcript_path}")
        
    try:
        with open(transcript_path, 'r') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading transcript file: {str(e)}")

def process_complete_transcript(audio_path, transcript_path, model_path):
    """Process a complete transcript file and classify all segments."""
    try:
        # Read the transcript file
        print(f"Reading transcript file: {transcript_path}")
        transcript_text = read_transcript_file(transcript_path)
        
        # Classify all segments
        print("Classifying segments...")
        results = classify_segments(audio_path, transcript_text, model_path)
        
        # Compare classifications with original labels
        comparison = compare_classifications(results)
        
        return comparison
        
    except Exception as e:
        raise RuntimeError(f"Error processing complete transcript: {str(e)}")

def compare_classifications(results):
    """Compare original speaker labels with model classifications."""
    total = len(results)
    matches = sum(1 for r in results if r['original_speaker'].lower() == r['classification'].lower())
    accuracy = matches / total if total > 0 else 0
    
    return {
        'total_segments': total,
        'matching_classifications': matches,
        'accuracy': accuracy,
        'detailed_results': results
    }

def save_results_to_csv(results, output_path):
    """Save classification results to a CSV file."""
    try:
        # Extract detailed results
        detailed_results = results['detailed_results']
        
        # Convert to DataFrame
        df = pd.DataFrame(detailed_results)
        
        # Reorder columns for better readability
        columns = ['original_speaker', 'classification', 'confidence', 
                  'start_time', 'end_time', 'median_pitch', 'pitch_mad']
        df = df[columns]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error saving results to CSV: {str(e)}")

if __name__ == "__main__":
    # File paths
    audio_path = "C:/Users/lukev/Projects/BehavBuddy/audio/leslie_voice.mp3"
    model_path = "C:/Users/lukev/Projects/BehavBuddy/trained_models/trained_leslie.pkl"
    transcript_path = "C:/Users/lukev/Projects/BehavBuddy/mp3TestingSet/transcriptionSet/processed-Rita + Mom 2.mp3.txt"  # Add your transcript path here
    output_path = "C:/Users/lukev/Projects/BehavBuddy/mp3TestingSet/testingFFTResults/results_leslie_voice.csv"  # Where to save the results
    
    try:
        # Print debug info
        print("\nChecking file paths:")
        print(f"Audio file exists: {os.path.exists(audio_path)}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        print(f"Transcript file exists: {os.path.exists(transcript_path)}")
        
        # Process the complete transcript
        print("\nProcessing complete transcript...")
        results = process_complete_transcript(audio_path, transcript_path, model_path)
        
        # Print summary
        print("\nClassification Summary:")
        print(f"Total segments analyzed: {results['total_segments']}")
        print(f"Matching classifications: {results['matching_classifications']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        
        # Save results to CSV
        save_results_to_csv(results, output_path)
        
        # Print detailed results
        print("\nDetailed Results:")
        for result in results['detailed_results']:
            print(f"\nSegment {result['start_time']:.2f}s - {result['end_time']:.2f}s:")
            print(f"Original speaker: {result['original_speaker']}")
            print(f"Classification: {result['classification']} (Confidence: {result['confidence']:.2%})")
            print(f"Median Pitch: {result['median_pitch']:.2f}")
            print(f"Pitch MAD: {result['pitch_mad']:.2f}")
    
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        # Print the full error traceback for debugging
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())