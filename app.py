import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import time
import shutil
import tempfile
from pathlib import Path
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import the audio grammar analysis modules
import warnings
import librosa
import re
import whisper
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Audio Grammar Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .file-list {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .audio-file {
        padding: 5px;
        margin: 2px 0;
        border-radius: 3px;
        cursor: pointer;
    }
    .audio-file:hover {
        background-color: #f0f2f5;
    }
    .audio-file.selected {
        background-color: #e6f0ff;
        border-left: 3px solid #1f77b4;
    }
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .analysis-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .score-gauge {
        text-align: center;
    }
    .upload-area {
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .folder-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 10px;
        background-color: #f0f2f5;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .folder-count {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 2px 8px;
        font-size: 12px;
    }
    .progress-container {
        margin-top: 20px;
    }
    .error-msg {
        color: #d62728;
        margin-top: 5px;
    }
    .success-msg {
        color: #2ca02c;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Simple sentence tokenizer
def simple_sent_tokenize(text):
    """Split text into sentences using basic rules"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# JSON encoder class to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Basic grammar rules checking class
class SimpleGrammarChecker:
    """Simple grammar checker using regex patterns"""

    def __init__(self):
        """Initialize grammar checking rules"""
        # Common grammar error patterns
        self.error_patterns = [
            # Subject-verb agreement errors
            (r'\b(I|we|they)\s+(is|was|has been)\b', "Subject-verb agreement error"),
            (r'\b(he|she|it)\s+(are|were|have been)\b', "Subject-verb agreement error"),
            (r'\bthe\s+(\w+s)\s+(\w+)s\b', "Possible subject-verb agreement error"),

            # Article errors
            (r'\ba\s+([aeiou]\w+)\b', "Article error (should be 'an' before vowel sounds)"),

            # Double negatives
            (r'\bnot\b.{1,20}\bno\b', "Double negative"),
            (r"\bdon't\b.{1,20}\bno\b", "Double negative"),
            (r"\bcan't\b.{1,20}\bnot\b", "Double negative"),

            # Common irregular verb errors
            (r'\bhave\s+went\b', "Irregular verb error (should be 'have gone')"),
            (r'\bhave\s+came\b', "Irregular verb error (should be 'have come')"),
            (r'\bhave\s+drank\b', "Irregular verb error (should be 'have drunk')"),
            (r'\bhas\s+went\b', "Irregular verb error (should be 'has gone')"),
            (r'\bhas\s+came\b', "Irregular verb error (should be 'has come')"),
            (r'\bhave\s+saw\b', "Irregular verb error (should be 'have seen')"),
            (r'\bhas\s+saw\b', "Irregular verb error (should be 'has seen')"),
            (r'\bwe\s+was\b', "Irregular verb error (should be 'we were')"),
            (r'\bthey\s+was\b', "Irregular verb error (should be 'they were')"),
            (r'\bI\s+seen\b', "Irregular verb error (should be 'I saw')"),
            (r'\bI\s+done\b', "Irregular verb error (should be 'I did')"),

            # Tense consistency
            (r'\byesterday\b.{1,30}\b(go|come|run|eat|speak|take)\b', "Tense error (past tense required)"),
            (r'\blast\s+(week|month|year)\b.{1,30}\b(go|come|run|eat|speak|take)\b', "Tense error (past tense required)"),

            # Redundant expressions
            (r'\b(and\s+also|but\s+however|past\s+history|future\s+plans|unexpected\s+surprise)\b', "Redundant expression"),

            # Preposition errors
            (r'\bdifferent\s+than\b', "Preposition error (should be 'different from')"),
            (r'\bin\s+regards\s+to\b', "Preposition error (should be 'with regard to')"),

            # Subject pronoun vs. object pronoun confusion
            (r'\bbetween\s+you\s+and\s+I\b', "Pronoun error (should be 'between you and me')"),
            (r'\blet\s+(I|we|they|he|she)\b', "Pronoun error (should use object form)"),

            # Common misspellings and errors
            (r'\balot\b', "Spelling error (should be 'a lot')"),
            (r'\beach\s+other\'s\b', "Possessive error"),
            (r'\byour\s+welcome\b', "Grammar error (should be 'you're welcome')"),
            (r'\bi\b', "Capitalization error (should be 'I')"),
            (r'\bthank\s+your\b', "Grammar error (should be 'thank you')"),

            # Misused words
            (r'\b(their|there|they\'re)\b', "Possible confusion of their/there/they're"),
            (r'\b(your|you\'re)\b', "Possible confusion of your/you're"),
            (r'\b(its|it\'s)\b', "Possible confusion of its/it's"),
            (r'\b(affect|effect)\b', "Possible confusion of affect/effect"),
            (r'\b(then|than)\b', "Possible confusion of then/than"),

            # Non-standard verb forms
            (r'\bgoes\s+to\b', "Potential verb form error"),
            (r'\bpeoples\b', "Non-standard plural (should be 'people')"),
            (r'\bchilds\b', "Non-standard plural (should be 'children')"),
            (r'\bdon\'t\s+\w+\b', "Possible contraction error"),

            # Word repetition
            (r'\b(\w+)\s+\1\b', "Word repetition")
        ]

    def check_grammar(self, text):
        """Check text for grammar errors using pattern matching"""
        if not text or text.strip() == '':
            return {
                'grammar_score': 0,
                'errors': [],
                'error_count': 0,
                'words': 0,
                'sentences': 0,
                'error_rate': 1.0  # 100% error rate for empty text
            }

        # Normalize text
        text = text.lower()

        # Count words and sentences
        sentences = simple_sent_tokenize(text)
        words = text.split()
        word_count = len(words)
        sentence_count = len(sentences)

        # Find grammar errors
        errors = []
        for pattern, message in self.error_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                error = {
                    'message': message,
                    'context': text[max(0, match.start() - 20):min(len(text), match.end() + 20)],
                    'pattern': pattern
                }
                errors.append(error)

        # Calculate error rate and grammar score
        error_count = len(errors)
        error_rate = error_count / max(1, word_count)

        # Scale score from 0-100, with fewer errors yielding higher scores
        # Using exponential decay for error rate to grammar score
        base_score = 100 * np.exp(-5 * error_rate)  # Steeper penalty for errors

        # Adjust score based on sentence complexity
        avg_words_per_sentence = word_count / max(1, sentence_count)
        complexity_factor = min(1.2, max(0.8, avg_words_per_sentence / 10))

        grammar_score = min(100, max(0, base_score * complexity_factor))

        return {
            'grammar_score': float(grammar_score),
            'errors': errors,
            'error_count': error_count,
            'words': word_count,
            'sentences': sentence_count,
            'error_rate': float(error_rate)
        }

# Whisper transcription class
class WhisperTranscriber:
    """Transcribes audio files to text using OpenAI's Whisper"""

    def __init__(self, model_size="base"):
        """Initialize Whisper model for transcription"""
        with st.spinner(f"Loading Whisper model ({model_size})..."):
            self.model = whisper.load_model(model_size)
            st.success("Whisper model loaded!")

    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using Whisper"""
        try:
            # Transcribe audio file
            result = self.model.transcribe(audio_file)

            # Extract transcription text
            text = result["text"]

            return {
                'success': True,
                'transcription': text,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'transcription': "",
                'error': f"Error transcribing audio: {str(e)}"
            }

# Modified AudioFeatureExtractor class with enhanced analysis
class AudioFeatureExtractor:
    """Extract audio features for analysis with basic text estimation"""

    def extract_audio_info(self, audio_file):
        """Extract audio features and attempt basic content estimation"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=None)

            # Basic features
            duration = librosa.get_duration(y=y, sr=sr)

            # Extract speech features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            zero_crossings = np.sum(librosa.zero_crossings(y))
            zero_crossing_rate = zero_crossings / len(y)

            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

            # Extract MFCCs (for phoneme analysis)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)

            # Calculate speech rate using onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Use the correct parameters format for peak_pick based on your librosa version
            peaks = librosa.util.peak_pick(
                onset_env,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.5,
                wait=10
            )
            speech_rate = len(peaks) / duration if duration > 0 else 0

            # Detect pauses (silent regions)
            intervals = librosa.effects.split(y, top_db=20)
            pauses = []
            if len(intervals) > 1:
                for i in range(len(intervals)-1):
                    pause_duration = (intervals[i+1][0] - intervals[i][1]) / sr
                    if pause_duration > 0.2:  # Consider pauses longer than 200ms
                        pauses.append(pause_duration)

            avg_pause_duration = np.mean(pauses) if pauses else 0
            pause_count = len(pauses)

            # Extract phrase count (approximate based on longer pauses)
            phrase_count = sum(1 for p in pauses if p > 0.5)

            # Basic segmentation based on silences
            segments = []
            for start, end in intervals:
                segment_duration = (end - start) / sr
                if segment_duration > 0.5:  # Skip very short segments
                    segments.append({
                        'start': float(start / sr),  # Convert numpy values to native Python types
                        'end': float(end / sr),
                        'duration': float(segment_duration)
                    })

            # Audio summary information
            audio_info = {
                'duration': float(duration),
                'tempo': float(tempo),
                'zero_crossing_rate': float(zero_crossing_rate),
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'spectral_rolloff': float(spectral_rolloff),
                'mfcc_means': mfcc_means.tolist(),  # Convert numpy array to list
                'speech_rate': float(speech_rate),
                'pause_count': int(pause_count),
                'avg_pause_duration': float(avg_pause_duration),
                'phrase_count': int(phrase_count),
                'segment_count': len(segments),
                'segments': segments
            }

            return audio_info

        except Exception as e:
            st.error(f"Error extracting audio information: {str(e)}")
            return None

# Direct Audio Grammar Analysis class
class AudioGrammarAnalyzer:
    """
    Analyzes audio patterns that often correlate with grammar quality
    without needing actual transcription
    """

    def analyze_audio_patterns(self, audio_info):
        """
        Analyze audio patterns that correlate with grammar quality
        Returns an estimation of grammar quality based on audio features
        """
        if not audio_info:
            return {
                'estimated_score': 50.0,  # Default midpoint score
                'confidence': 'very low',
                'audio_analysis': ['No audio features available for analysis']
            }

        # Audio features that correlate with grammar quality
        speech_rate = audio_info.get('speech_rate', 0)
        pause_count = audio_info.get('pause_count', 0)
        avg_pause_duration = audio_info.get('avg_pause_duration', 0)
        phrase_count = audio_info.get('phrase_count', 0)
        duration = audio_info.get('duration', 0)

        # Calculate pause-to-speech ratio
        total_pause_time = avg_pause_duration * pause_count
        speech_time = max(0.1, duration - total_pause_time)
        pause_speech_ratio = total_pause_time / speech_time if speech_time > 0 else 0

        # Calculate speech fluency score
        # More regular pauses and moderate speech rate often correlate with better grammar
        fluency_score = 0

        # Speech rate component (penalize extremely fast or slow speech)
        if 2.0 <= speech_rate <= 3.5:
            fluency_score += 30  # Optimal speech rate
        elif 1.5 <= speech_rate < 2.0 or 3.5 < speech_rate <= 4.0:
            fluency_score += 25  # Good speech rate
        elif 1.0 <= speech_rate < 1.5 or 4.0 < speech_rate <= 4.5:
            fluency_score += 20  # Acceptable speech rate
        else:
            fluency_score += 15  # Poor speech rate

        # Pause distribution component
        if duration > 0:
            pauses_per_minute = pause_count / (duration / 60)
            if 7 <= pauses_per_minute <= 15:
                fluency_score += 30  # Optimal pause frequency
            elif 5 <= pauses_per_minute < 7 or 15 < pauses_per_minute <= 20:
                fluency_score += 25  # Good pause frequency
            elif 3 <= pauses_per_minute < 5 or 20 < pauses_per_minute <= 25:
                fluency_score += 20  # Acceptable pause frequency
            else:
                fluency_score += 15  # Poor pause frequency

        # Phrase structure component (based on phrase count)
        if duration > 0:
            phrases_per_minute = phrase_count / (duration / 60)
            if 8 <= phrases_per_minute <= 16:
                fluency_score += 40  # Optimal phrase structure
            elif 6 <= phrases_per_minute < 8 or 16 < phrases_per_minute <= 20:
                fluency_score += 35  # Good phrase structure
            elif 4 <= phrases_per_minute < 6 or 20 < phrases_per_minute <= 25:
                fluency_score += 30  # Acceptable phrase structure
            else:
                fluency_score += 25  # Poor phrase structure

        # Generate analysis text based on features
        analysis_points = []

        if speech_rate > 4.0:
            analysis_points.append("Speech rate is very fast, which may indicate rushed speech with potential grammar issues.")
        elif speech_rate < 1.5 and speech_rate > 0:
            analysis_points.append("Speech rate is quite slow, which may indicate deliberate speech but could also suggest hesitation.")
        else:
            analysis_points.append("Speech rate is moderate, suggesting comfortable language fluency.")

        if pause_count < 3 and duration > 20:
            analysis_points.append("Very few pauses detected, which may indicate run-on sentences or limited sentence structure.")
        elif pause_count > 25 and duration < 60:
            analysis_points.append("Many short pauses detected, which may indicate choppy speech or hesitation.")

        if phrase_count < 3 and duration > 20:
            analysis_points.append("Limited phrase structure detected, which may indicate simple grammatical constructions.")
        elif phrase_count > 15 and duration < 30:
            analysis_points.append("Complex phrase structure detected, suggesting varied grammatical constructions.")

        # Determine confidence level based on audio quality and duration
        confidence = 'low'
        if duration > 30 and audio_info.get('segment_count', 0) > 5:
            confidence = 'medium'
        if duration > 60 and audio_info.get('segment_count', 0) > 10:
            confidence = 'fair'

        return {
            'estimated_score': float(fluency_score),  # Ensure it's a Python float
            'confidence': confidence,
            'audio_analysis': analysis_points
        }

# Whisper-based grammar scoring class
class WhisperGrammarScorer:
    def __init__(self, whisper_model_size="base"):
        """Initialize the Whisper-based Grammar Scoring Engine"""
        st.info("Initializing Whisper-based Grammar Scoring Engine...")

        # Initialize components
        self.audio_extractor = AudioFeatureExtractor()
        self.audio_analyzer = AudioGrammarAnalyzer()
        self.transcriber = WhisperTranscriber(model_size=whisper_model_size)
        self.grammar_checker = SimpleGrammarChecker()

        st.success("Initialization complete!")

    def process_audio(self, audio_file):
        """Process a single audio file and analyze its grammar"""
        st.info(f"Processing audio: {os.path.basename(audio_file)}")

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract audio information
        status_text.text("Extracting audio features...")
        progress_bar.progress(10)
        audio_info = self.audio_extractor.extract_audio_info(audio_file)
        progress_bar.progress(30)

        # Analyze audio patterns for fluency estimation
        status_text.text("Analyzing audio patterns...")
        audio_analysis = self.audio_analyzer.analyze_audio_patterns(audio_info)
        progress_bar.progress(50)

        # Try to transcribe the audio
        status_text.text("Transcribing audio with Whisper...")
        transcription_result = self.transcriber.transcribe_audio(audio_file)
        progress_bar.progress(80)

        # Initialize text analysis as empty
        text_analysis = {
            'grammar_score': 0,
            'errors': [],
            'error_count': 0,
            'words': 0,
            'sentences': 0,
            'error_rate': 1.0
        }

        # If transcription is successful, analyze grammar
        status_text.text("Analyzing grammar...")
        if transcription_result['success']:
            text = transcription_result['transcription']
            text_analysis = self.grammar_checker.check_grammar(text)

        progress_bar.progress(90)

        # Calculate combined score - weighted average of audio and text scores
        # If transcription failed, rely solely on audio analysis
        if transcription_result['success']:
            # Weight text analysis higher (70%) than audio analysis (30%)
            combined_score = 0.3 * audio_analysis['estimated_score'] + 0.7 * text_analysis['grammar_score']
            confidence = 'high' if audio_analysis['confidence'] != 'low' else 'medium'
        else:
            combined_score = audio_analysis['estimated_score']
            confidence = audio_analysis['confidence']

        progress_bar.progress(100)
        status_text.text("Analysis complete!")

        # Combine results
        result = {
            'audio_file': audio_file,
            'audio_info': {k: v for k, v in audio_info.items() if k != 'mfcc_means' and k != 'segments'} if audio_info else None,
            'audio_analysis': audio_analysis,
            'transcription': {
                'success': transcription_result['success'],
                'text': transcription_result['transcription'] if transcription_result['success'] else "",
                'error': transcription_result['error']
            },
            'text_analysis': text_analysis if transcription_result['success'] else None,
            'combined_results': {
                'score': float(combined_score),
                'confidence': confidence
            }
        }

        return result

    def batch_process(self, audio_files):
        """Process multiple audio files"""
        results = []

        # Process each file
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, file_path in enumerate(audio_files):
            progress_text.text(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
            progress_bar.progress(i / len(audio_files))
            
            try:
                with st.expander(f"Processing {os.path.basename(file_path)}", expanded=False):
                    result = self.process_audio(file_path)
                    results.append(result)
            except Exception as e:
                st.error(f"Error processing {file_path}: {str(e)}")
                
        progress_bar.progress(1.0)
        progress_text.text("All files processed!")
            
        return results

    def visualize_results(self, results):
        """Create visualizations for audio grammar analysis results"""
        if not results:
            st.warning("No results to visualize")
            return None

        # Extract data
        df_data = []

        for result in results:
            filename = os.path.basename(result['audio_file'])

            # Get combined score
            combined_score = result['combined_results']['score']
            confidence = result['combined_results']['confidence']

            # Get audio features
            audio_info = result.get('audio_info', {})
            if audio_info:
                duration = audio_info.get('duration', 0)
                speech_rate = audio_info.get('speech_rate', 0)
                pause_count = audio_info.get('pause_count', 0)
                phrase_count = audio_info.get('phrase_count', 0)
            else:
                duration = 0
                speech_rate = 0
                pause_count = 0
                phrase_count = 0

            # Get transcription and text analysis info
            transcription_success = result['transcription']['success']
            text = result['transcription']['text'] if transcription_success else ""

            # Get text analysis metrics if available
            if transcription_success and result['text_analysis']:
                text_score = result['text_analysis']['grammar_score']
                error_count = result['text_analysis']['error_count']
                word_count = result['text_analysis']['words']
                sentence_count = result['text_analysis']['sentences']
                error_rate = result['text_analysis']['error_rate']
            else:
                text_score = 0
                error_count = 0
                word_count = 0
                sentence_count = 0
                error_rate = 0

            # Audio analysis score
            audio_score = result['audio_analysis']['estimated_score']

            row = {
                'Filename': filename,
                'Combined Score': combined_score,
                'Audio Score': audio_score,
                'Text Score': text_score if transcription_success else np.nan,
                'Confidence': confidence,
                'Duration': duration,
                'Speech Rate': speech_rate,
                'Pause Count': pause_count,
                'Phrase Count': phrase_count,
                'Transcription Success': transcription_success,
                'Word Count': word_count,
                'Sentence Count': sentence_count,
                'Error Count': error_count,
                'Error Rate': error_rate
            }
            df_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        return df

# Function to render audio file list
def render_audio_files(folder_path, selected_file=None):
    st.markdown("""
    <div class="folder-header">
        <div><b>📁 Audio Files</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    if os.path.exists(folder_path):
        audio_files = [
            f for f in os.listdir(folder_path) 
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        
        if not audio_files:
            st.markdown("""
            <div class="file-list">
                <p>No audio files found. Upload some files to get started.</p>
            </div>
            """, unsafe_allow_html=True)
            return [], None
        
        st.markdown(f"""
        <div class="folder-header">
            <div>Files in folder:</div>
            <div class="folder-count">{len(audio_files)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="file-list">', unsafe_allow_html=True)
        
        for file in sorted(audio_files):
            file_path = os.path.join(folder_path, file)
            selected_class = "selected" if file == selected_file else ""
            
            # Create a clickable div for each file
            if st.button(f"🔊 {file}", key=f"file_{file}"):
                selected_file = file
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate full paths
        full_paths = [os.path.join(folder_path, f) for f in audio_files]
        
        return full_paths, selected_file
    else:
        st.warning(f"Folder {folder_path} not found. Please create it or choose another folder.")
        return [], None

def create_score_gauge(score, title="Grammar Score"):
    """Create a gauge chart for displaying scores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_score_color(score)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(255, 165, 0, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(0, 128, 0, 0.2)'}
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
    )
    
    return fig

def get_score_color(score):
    """Return color based on score"""
    if score < 30:
        return "red"
    elif score < 70:
        return "orange"
    else:
        return "green"

def display_transcription_with_errors(transcription, errors):
    """Display transcription text with highlighted errors"""
    if not transcription or not errors:
        return transcription
    
    # Create HTML with highlighted errors
    html_text = transcription
    for error in errors:
        context = error.get('context', '')
        if context:
            html_text = html_text.replace(
                context,
                f'<span style="background-color: rgba(255,0,0,0.2);" title="{error["message"]}">{context}</span>'
            )
    
    return html_text

def get_audio_file_player(file_path):
    """Get HTML for audio player"""
    audio_format = file_path.split('.')[-1].lower()
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls style="width: 100%;">
      <source src="data:audio/{audio_format};base64,{audio_b64}" type="audio/{audio_format}">
      Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def handle_file_upload(upload_folder):
    """Handle file uploads to specified folder"""
    uploaded_files = st.file_uploader("Upload audio files for analysis", 
                                     type=["wav", "mp3", "flac", "ogg", "m4a"],
                                     accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Create directory if it doesn't exist
            os.makedirs(upload_folder, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(upload_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
        st.success(f"{len(uploaded_files)} files uploaded successfully!")
        # Force a rerun to update the file list
        st.rerun()

def main():
    # Create app header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #1f77b4;">🎙️ Audio Grammar Analyzer</h1>
        <p style="font-size: 18px;">Upload and analyze audio files for grammar quality assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## Settings")
        
        # Set audio folder
        audio_folder = st.text_input("Audio Folder Path", "audio_files", 
                                     help="Path to the folder containing audio files")
        
        # Create folder if it doesn't exist
        if not os.path.exists(audio_folder):
            if st.button("Create Folder"):
                os.makedirs(audio_folder, exist_ok=True)
                st.success(f"Folder '{audio_folder}' created!")
        
        # Upload interface
        st.markdown("## Upload Files")
        handle_file_upload(audio_folder)
        
        # Whisper model selection
        st.markdown("## Whisper Model")
        whisper_model = st.selectbox(
            "Select Whisper model size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display audio files
        audio_files, selected_file = render_audio_files(audio_folder)
        
        # Add batch processing button
        if audio_files:
            st.markdown("### Batch Processing")
            if st.button("🔍 Analyze All Files"):
                # Initialize grammar scorer
                scorer = WhisperGrammarScorer(whisper_model_size=whisper_model)
                
                # Process all files
                results = scorer.batch_process(audio_files)
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.analysis_df = scorer.visualize_results(results)
                
                st.success("Batch analysis complete!")
    
    with col2:
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            # Display analysis dashboard
            st.markdown("## 📊 Analysis Dashboard")
            
            # Get DataFrame of results
            df = st.session_state.analysis_df
            
            # Create dashboard metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="metric-label">Average Grammar Score</div>
                    <div class="metric-value">{:.1f}</div>
                </div>
                """.format(df['Combined Score'].mean()), unsafe_allow_html=True)
                
            with col_b:
                st.markdown("""
                <div class="dashboard-card">
                    <div class="metric-label">Files Analyzed</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
                
            with col_c:
                transcription_success_rate = df['Transcription Success'].mean() * 100
                st.markdown("""
                <div class="dashboard-card">
                    <div class="metric-label">Transcription Success Rate</div>
                    <div class="metric-value">{:.1f}%</div>
                </div>
                """.format(transcription_success_rate), unsafe_allow_html=True)
            
            # Score distribution
            st.markdown("### Score Distribution")
            fig = px.histogram(df, x='Combined Score', nbins=10, 
                               title="Distribution of Grammar Scores",
                               color_discrete_sequence=['#1f77b4'])
            fig.update_layout(xaxis_title="Grammar Score", yaxis_title="Number of Files")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table of results
            st.markdown("### Results Table")
            
            # Format the table
            display_df = df[['Filename', 'Combined Score', 'Audio Score', 'Text Score', 
                             'Duration', 'Error Count', 'Word Count']].copy()
            
            # Format numbers
            display_df['Combined Score'] = display_df['Combined Score'].round(1)
            display_df['Audio Score'] = display_df['Audio Score'].round(1)
            display_df['Text Score'] = display_df['Text Score'].round(1)
            display_df['Duration'] = display_df['Duration'].round(1)
            
            # Sort by score
            display_df = display_df.sort_values('Combined Score', ascending=False)
            
            # Display table
            st.dataframe(display_df, use_container_width=True)
            
            # Detailed file view
            st.markdown("### Detailed File Analysis")
            selected_result_file = st.selectbox("Select a file to view details", 
                                               df['Filename'].tolist())
            
            # Get selected result
            selected_result = next((r for r in st.session_state.analysis_results 
                                  if os.path.basename(r['audio_file']) == selected_result_file), None)
            
            if selected_result:
                # Display results in tabs
                tabs = st.tabs(["Overview", "Transcription", "Grammar Analysis", "Audio Analysis"])
                
                with tabs[0]:
                    # Overview tab
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        # Display score gauge
                        st.markdown("#### Overall Grammar Score")
                        fig = create_score_gauge(selected_result['combined_results']['score'], 
                                              "Grammar Score")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence
                        st.markdown(f"**Confidence**: {selected_result['combined_results']['confidence'].capitalize()}")
                        
                    with col_b:
                        # Audio player
                        st.markdown("#### Listen to Audio")
                        st.audio(selected_result['audio_file'])
                        
                        # Basic info
                        duration = selected_result['audio_info']['duration'] if selected_result['audio_info'] else 0
                        st.markdown(f"**Duration**: {duration:.1f} seconds")
                        
                        # Transcription status
                        if selected_result['transcription']['success']:
                            st.markdown("✅ **Transcription**: Successful")
                        else:
                            st.markdown("❌ **Transcription**: Failed")
                            st.markdown(f"Error: {selected_result['transcription']['error']}")
                
                with tabs[1]:
                    # Transcription tab
                    if selected_result['transcription']['success']:
                        st.markdown("#### Audio Transcription")
                        
                        # Display text with errors highlighted if available
                        if 'text_analysis' in selected_result and selected_result['text_analysis']:
                            html_text = display_transcription_with_errors(
                                selected_result['transcription']['text'],
                                selected_result['text_analysis']['errors']
                            )
                            st.markdown(f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 5px;'>{html_text}</div>", 
                                       unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 5px;'>{selected_result['transcription']['text']}</div>", 
                                       unsafe_allow_html=True)
                    else:
                        st.warning("Transcription failed for this audio file.")
                        st.markdown(f"Error: {selected_result['transcription']['error']}")
                
                with tabs[2]:
                    # Grammar analysis tab
                    if 'text_analysis' in selected_result and selected_result['text_analysis']:
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            st.markdown("#### Grammar Score")
                            fig = create_score_gauge(selected_result['text_analysis']['grammar_score'], 
                                                  "Text Grammar")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_b:
                            st.markdown("#### Text Stats")
                            text_analysis = selected_result['text_analysis']
                            
                            st.markdown(f"""
                            - **Words**: {text_analysis['words']}
                            - **Sentences**: {text_analysis['sentences']}
                            - **Grammar Errors**: {text_analysis['error_count']}
                            - **Error Rate**: {text_analysis['error_rate']:.3f} errors per word
                            """)
                        
                        # Grammar errors
                        if text_analysis['errors']:
                            st.markdown("#### Grammar Errors")
                            
                            for i, error in enumerate(text_analysis['errors']):
                                with st.expander(f"Error {i+1}: {error['message']}"):
                                    st.markdown(f"**Context**: \"...{error['context']}...\"")
                        else:
                            st.success("No grammar errors detected!")
                    else:
                        st.warning("Grammar analysis not available for this file.")
                
                with tabs[3]:
                    # Audio analysis tab
                    if 'audio_info' in selected_result and selected_result['audio_info']:
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            st.markdown("#### Audio Fluency Score")
                            fig = create_score_gauge(selected_result['audio_analysis']['estimated_score'], 
                                                  "Audio Fluency")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_b:
                            st.markdown("#### Audio Metrics")
                            audio_info = selected_result['audio_info']
                            
                            st.markdown(f"""
                            - **Duration**: {audio_info['duration']:.1f} seconds
                            - **Speech Rate**: {audio_info['speech_rate']:.2f} syllables/second
                            - **Pauses**: {audio_info['pause_count']} (avg: {audio_info['avg_pause_duration']:.2f}s)
                            - **Phrases**: {audio_info['phrase_count']}
                            """)
                        
                        # Audio analysis insights
                        st.markdown("#### Audio Analysis Insights")
                        for insight in selected_result['audio_analysis']['audio_analysis']:
                            st.markdown(f"- {insight}")
                    else:
                        st.warning("Audio analysis not available for this file.")
        
        # If no files have been analyzed yet
        elif audio_files:
            st.markdown("## 🎙️ Audio Grammar Analysis")
            st.info("Select 'Analyze All Files' in the left panel to start batch processing, or upload new files and refresh after uploading new file!.")
            
            # Display empty state image or message
            st.markdown("""
            <div style="text-align: center; margin-top: 100px; color: #aaa;">
                <span style="font-size: 72px;">🎙️</span>
                <p style="font-size: 24px;">Ready to analyze your audio files</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("## 🎙️ Audio Grammar Analysis")
            st.warning(f"No audio files found in folder: {audio_folder}")
            st.markdown("""
            <div class="upload-area">
                <span style="font-size: 48px;">📤</span>
                <p>Upload audio files using the sidebar to get started</p>
            </div>
            """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()