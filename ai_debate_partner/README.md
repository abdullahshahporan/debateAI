# AI Debate Partner: A Voice-Based Debate Evaluation System Using OpenAI API and Fuzzy Logic

## Objective
This project is an intelligent Python-based debate practice system. A user speaks an argument on any chosen debate topic, the system converts the speech to text using OpenAI Speech-to-Text, analyzes the transcript, evaluates it with a fuzzy inference system, and then returns a final score, classification, personalized feedback, and an AI-generated counter-argument.

## Features
- User enters any debate topic.
- AI side can be assigned randomly, or the user can choose For/Against and let the AI take the opposite side.
- Voice input through a centered microphone button.
- Automatic transcription and evaluation after a microphone recording is stopped.
- OpenAI Speech-to-Text transcription.
- NLP-based preprocessing and feature extraction.
- Rule-based fallacy detection.
- Sentiment-based emotional bias estimation.
- Fuzzy logic evaluation of debate performance.
- OpenAI-generated counter-argument.
- OpenAI-generated personalized feedback.
- Clean Streamlit dashboard with transcript preview on the right side.
- Bar chart visualization of all evaluation criteria.

## Technology Used
- Python 3.x
- OpenAI API
- scikit-fuzzy
- NLTK or simple regex-based preprocessing
- TextBlob
- re
- time / timing logic
- Streamlit
- matplotlib
- python-dotenv

## System Workflow
1. User selects a debate topic.
2. System randomly assigns AI as For or Against.
3. User gets the opposite side.
4. User records a voice response within the selected time limit.
5. When recording stops, the audio is sent to OpenAI Speech-to-Text automatically.
6. The transcript is preprocessed and analyzed.
7. Numerical features are extracted from the argument.
8. The fuzzy inference system calculates the final debate score.
9. The system classifies the user as Beginner, Intermediate, or Advanced Debater.
10. OpenAI generates a counter-argument and personalized feedback.

## Fuzzy Logic Explanation
The project uses fuzzy logic because debate quality is not always binary. A response can be partially logical, partially evidence-based, or moderately clear. Fuzzy inference is appropriate for handling uncertainty and combining multiple subjective qualities into a final score.

## Input and Output
### Input variables
- logical_strength
- evidence_usage
- clarity
- emotional_bias
- fallacy_level
- time_efficiency
- relevance

### Output variable
- debate_score

### Classification rules
- 0.0 to 4.0 = Beginner Debater
- 4.1 to 7.0 = Intermediate Debater
- 7.1 to 10.0 = Advanced Debater

## Installation Steps
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`.
4. Add your OpenAI API key to `.env`.

## How to Set OpenAI API Key
Create a `.env` file in the project folder and add:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_STT_MODEL=whisper-1
```

## How to Run the Project
From the project directory:
```bash
streamlit run app.py
```

## Example Output
- Transcript: generated from the user's voice.
- Logical Strength: 7.5/10
- Evidence Usage: 6.0/10
- Clarity: 7.0/10
- Emotional Bias: 2.0/10
- Fallacy Level: 1.0/10
- Time Efficiency: 8.0/10
- Relevance: 7.5/10
- Final Fuzzy Debate Score: 7.2/10
- Classification: Intermediate Debater
- AI Counter-Argument: a concise opposing argument.
- Personalized Feedback: strengths, weaknesses, improvement suggestions, and a better sample argument.

## Future Improvements
- Add live streaming partial transcripts while the user is still speaking.
- Add speaker diarization and pronunciation feedback.
- Add debate history and progress tracking.
- Add richer fallacy detection using a fine-tuned NLP model.
- Add more nuanced scoring categories for persuasion and rebuttal quality.

## Deployment Note
See [DEPLOYMENT.md](DEPLOYMENT.md) for Vercel-specific guidance and recommended hosting options for this Streamlit-based project.

## Viva / Presentation Note
This project uses OpenAI API for speech-to-text conversion and natural language feedback generation. Fuzzy logic is used for debate evaluation because argument quality is not always binary. A response can be partially logical, partially evidence-based, or moderately clear. Therefore, fuzzy inference is suitable for handling uncertainty in debate performance evaluation.
