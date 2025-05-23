# -*- coding: utf-8 -*-
"""shl.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZsU7PbU_xqQU2X7oIpIR-LF9Z1vSUk3-
"""

# For grammar analysis
!pip install language-tool-python
!pip install transformers
!pip install evaluate
!pip install sacrebleu
!pip install nltk

# For error analysis
!pip install spacy
!python -m spacy download en_core_web_sm

!pip install transformers datasets torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

!apt-get update && apt-get install -y openjdk-11-jdk
!pip install language_tool_python

!pip install -q openai-whisper
!pip install -q librosa matplotlib seaborn

"""**ALL IMPORTS**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
import warnings
import librosa
import librosa.display
import whisper
from IPython.display import HTML, display, Audio
from google.colab import files
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

"""**TOKANIZATION**"""

def simple_sent_tokenize(text):
    """Split text into sentences using basic rules"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

"""**WHISPER TRANSCRIPTION CLASS**"""

class WhisperTranscriber:
    """Transcribes audio files to text using OpenAI's Whisper"""

    def __init__(self, model_size="base"):
        """Initialize Whisper model for transcription"""
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded!")

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

"""**NUMPY ENCODER & SIMPLE GRAMMAR CHECKER**"""

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

"""**AUDIO FEATURE EXTRACTION**"""

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
            print(f"Error extracting audio information: {str(e)}")
            return None

"""**AUDIO GRAMMMAR CHECKER**"""

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

"""**COMBINING BOTH (AUDIO & GRAMMAR) (HYBRID MODEL)**"""

class WhisperGrammarScorer:
    def __init__(self, whisper_model_size="base"):
        """Initialize the Whisper-based Grammar Scoring Engine"""
        print("Initializing Whisper-based Grammar Scoring Engine...")

        # Initialize components
        self.audio_extractor = AudioFeatureExtractor()
        self.audio_analyzer = AudioGrammarAnalyzer()
        self.transcriber = WhisperTranscriber(model_size=whisper_model_size)
        self.grammar_checker = SimpleGrammarChecker()

        print("Initialization complete!")

    def process_audio(self, audio_file):
        """Process a single audio file and analyze its grammar"""
        print(f"Processing audio: {audio_file}")

        # Extract audio information
        audio_info = self.audio_extractor.extract_audio_info(audio_file)

        # Analyze audio patterns for fluency estimation
        audio_analysis = self.audio_analyzer.analyze_audio_patterns(audio_info)

        # Try to transcribe the audio
        transcription_result = self.transcriber.transcribe_audio(audio_file)

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
        if transcription_result['success']:
            text = transcription_result['transcription']
            text_analysis = self.grammar_checker.check_grammar(text)

        # Calculate combined score - weighted average of audio and text scores
        # If transcription failed, rely solely on audio analysis
        if transcription_result['success']:
            # Weight text analysis higher (70%) than audio analysis (30%)
            combined_score = 0.3 * audio_analysis['estimated_score'] + 0.7 * text_analysis['grammar_score']
            confidence = 'high' if audio_analysis['confidence'] != 'low' else 'medium'
        else:
            combined_score = audio_analysis['estimated_score']
            confidence = audio_analysis['confidence']

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

    def batch_process(self, audio_directory, output_file=None, limit=None):
        """Process multiple audio files"""
        results = []

        # Check if directory exists
        if not os.path.exists(audio_directory):
            print(f"Audio directory not found: {audio_directory}")
            return results

        # Get all valid audio files
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = [
            os.path.join(audio_directory, file)
            for file in os.listdir(audio_directory)
            if any(file.lower().endswith(ext) for ext in valid_extensions)
        ]

        if not audio_files:
            print(f"No audio files found in: {audio_directory}")
            return results

        # Apply limit if specified
        if limit and limit > 0:
            audio_files = audio_files[:limit]
            print(f"Processing limited set of {len(audio_files)} files")

        # Process each file
        for file_path in tqdm(audio_files, desc="Processing audio files"):
            try:
                result = self.process_audio(file_path)
                results.append(result)

                # Display interim results
                filename = os.path.basename(file_path)
                print(f"File: {filename}")

                # Show combined score if available
                combined_score = result['combined_results']['score']
                confidence = result['combined_results']['confidence']
                print(f"Combined Grammar Score: {combined_score:.1f} (confidence: {confidence})")

                # Show transcription status
                if result['transcription']['success']:
                    print(f"Transcription: \"{result['transcription']['text'][:100]}...\"")
                    print(f"Grammar Errors: {result['text_analysis']['error_count']}")
                else:
                    print(f"Transcription failed: {result['transcription']['error']}")
                    print(f"Using audio-only score: {result['audio_analysis']['estimated_score']:.1f}")

                # Show audio analysis
                print(f"Audio Analysis: {', '.join(result['audio_analysis']['audio_analysis'])}")
                print("-" * 40)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        # Save results if requested
        if output_file and results:
            # Use the custom JSON encoder to handle NumPy types
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            print(f"Results saved to {output_file}")

        return results

    def visualize_results(self, results):
        """Create visualizations for audio grammar analysis results"""
        if not results:
            print("No results to visualize")
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

        # Visualizations

        # 1. Combined score distribution histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Combined Score'], bins=10, kde=True)
        plt.title('Distribution of Combined Grammar Scores', fontsize=16)
        plt.xlabel('Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()

        # 2. Individual scores bar chart (if <= 20 files)
        if len(df) <= 20:
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='Filename', y='Combined Score', data=df)
            plt.title('Combined Grammar Scores by Audio File', fontsize=16)
            plt.xlabel('Audio File', fontsize=14)
            plt.ylabel('Score', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3, axis='y')

            # Add score labels
            for i, p in enumerate(ax.patches):
                ax.annotate(f"{p.get_height():.1f}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

        # 3. Compare audio score vs text score (for successful transcriptions)
        transcribed_df = df[df['Transcription Success'] == True].copy()
        if len(transcribed_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(transcribed_df['Audio Score'], transcribed_df['Text Score'], s=100, alpha=0.7)

            # Add file labels
            for i, row in transcribed_df.iterrows():
                plt.annotate(row['Filename'],
                            (row['Audio Score'], row['Text Score']),
                            xytext=(5, 5), textcoords='offset points')

            # Add diagonal line for reference
            max_val = max(transcribed_df['Audio Score'].max(), transcribed_df['Text Score'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

            plt.title('Audio Score vs Text Grammar Score', fontsize=16)
            plt.xlabel('Audio Analysis Score', fontsize=14)
            plt.ylabel('Text Grammar Score', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.show()

        # 4. Error rate vs combined score
        if len(transcribed_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(transcribed_df['Error Rate'], transcribed_df['Combined Score'], s=100, alpha=0.7)

            # Add trendline
            z = np.polyfit(transcribed_df['Error Rate'], transcribed_df['Combined Score'], 1)
            p = np.poly1d(z)
            plt.plot(transcribed_df['Error Rate'], p(transcribed_df['Error Rate']), "r--", alpha=0.7)

            plt.title('Grammar Error Rate vs Combined Score', fontsize=16)
            plt.xlabel('Error Rate (errors/word)', fontsize=14)
            plt.ylabel('Combined Score', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.show()

        # Return the DataFrame for further analysis
        return df

"""**UPLOAD & ANALYSIS**"""

def upload_and_analyze_audio(whisper_model_size="base"):
    """Upload audio files and analyze them"""

    print("Please upload audio files for grammar analysis...")

    # Upload audio files
    uploaded = files.upload()

    if not uploaded:
        print("No files were uploaded.")
        return None

    # Create a temporary directory to store the uploaded files
    temp_dir = "uploaded_audio"
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded files to the temporary directory
    for filename, content in uploaded.items():
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content)

    # Initialize the grammar scorer
    scorer = WhisperGrammarScorer(whisper_model_size=whisper_model_size)

    # Process the uploaded files
    results = scorer.batch_process(temp_dir, output_file="whisper_grammar_results.json")

    # Visualize the results
    df = scorer.visualize_results(results)

    return results

"""**MAIN EXECUTION**"""

print("Whisper-based Audio Grammar Scoring Engine")
print("----------------------------------------------------")
print("FULL VERSION - Transcribes and analyzes grammar using Whisper")
print("----------------------------------------------------")
scorer = WhisperGrammarScorer(whisper_model_size="base")
results = scorer.batch_process("/content/drive/MyDrive/audio_shl")
df = scorer.visualize_results(results)

