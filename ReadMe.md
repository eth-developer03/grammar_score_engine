# 🎙️ Audio Grammar Analyzer

An advanced audio analysis tool that uses AI to evaluate grammar quality in speech recordings. This project uses OpenAI's Whisper model for speech-to-text transcription, combined with custom grammar analysis algorithms to provide comprehensive assessments of spoken language.

**Live Demo**: [Try Audio Grammar Analyzer on Streamlit Cloud](https://grammarscoreengine-shl.streamlit.app/)

## 🚀 Features

- **Speech-to-Text Transcription**: Uses OpenAI's Whisper model for accurate audio transcription
- **Grammar Analysis**: Evaluates grammar quality based on recognized text patterns
- **Audio Pattern Analysis**: Analyzes speech patterns, pauses, and fluency metrics
- **Combined Scoring**: Provides a comprehensive grammar quality score
- **Detailed Error Detection**: Identifies specific grammar issues in transcribed text
- **Batch Processing**: Process multiple audio files at once
- **Interactive Dashboard**: Visualize results with intuitive charts and gauges
- **User-Friendly Interface**: Easy file management and analysis

## 📋 How It Works

1. **Audio Upload**: Add audio files through the UI or place them in a designated folder
2. **Feature Extraction**: Extract speech patterns, pauses, and other audio metrics
3. **Transcription**: Convert speech to text using Whisper AI
4. **Grammar Analysis**: Analyze transcribed text for grammar errors
5. **Fluency Analysis**: Evaluate speech patterns for fluency indicators
6. **Combined Scoring**: Generate a comprehensive score based on text and audio analysis
7. **Results Visualization**: Display detailed analysis with interactive charts

## 🖥️ Getting Started

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Used Conda for env creation

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/eth-developer03/grammar_score_engine

   ```

2. Install required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 🎮 Usage Instructions

### File Management

- **Upload Files**: Use the sidebar uploader to add audio files
- **Create Folders**: Create new folders for organizing your audio files
- **Browse Files**: Navigate through your audio collection in the sidebar

### Analysis

- **Single File**: Select any file from the list to analyze it individually
- **Batch Processing**: Click "Analyze All Files" to process your entire collection
- **Configuration**: Select Whisper model size based on your needs and hardware capabilities

### Results

- **Dashboard**: View overall statistics for your audio collection
- **File Details**: Explore detailed analysis for each audio file:
  - Grammar score
  - Identified errors
  - Speech metrics
  - Transcription text
  - Audio insights

## 📊 Analysis Metrics

### Text Analysis

- **Grammar Score**: Overall grammar quality score (0-100)
- **Error Rate**: Number of errors per word
- **Word Count**: Total words in the transcription
- **Sentence Count**: Number of sentences detected
- **Error Types**: Categories of grammar errors identified

### Audio Analysis

- **Fluency Score**: Speech fluency quality (0-100)
- **Speech Rate**: Words/syllables per minute
- **Pause Distribution**: Analysis of pauses in speech
- **Phrase Structure**: Complexity and variety of spoken phrases

## 🧪 Technical Details

### Components

- **WhisperTranscriber**: Handles speech-to-text conversion
- **SimpleGrammarChecker**: Analyzes text for grammar errors
- **AudioFeatureExtractor**: Extracts features from audio signals
- **AudioGrammarAnalyzer**: Analyzes audio patterns related to grammar quality
- **WhisperGrammarScorer**: Combines all components for comprehensive scoring

### Models

- **Whisper Models**:
  - tiny: Fastest, lowest accuracy
  - base: Good balance of speed and accuracy
  - small: Better accuracy, slower processing
  - medium: High accuracy, slower processing
  - large: Highest accuracy, slowest processing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## [_LINK OF THE IPYNB FILE_](https://colab.research.google.com/drive/1ZsU7PbU_xqQU2X7oIpIR-LF9Z1vSUk3-?usp=sharing)
