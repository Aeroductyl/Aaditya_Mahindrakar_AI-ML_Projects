import openai
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi 
import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# API keys
youtube_api_key = 'AIzaSyBHqQHZkq2CkN8Ctn4dVZyHbcjVQRAkp2I'
openai_api_key = 'sk-proj-HcdgvZcSST1aY1N4AGY2T3BlbkFJBl3kghLzYsnVtHFsy7xs'

# Authentication of YouTube API
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

# Function to search videos
def search_videos(query, max_results=3):
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    )
    response = request.execute()
    return response['items']


# Function to get transcription
def get_transcription(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([t['text'] for t in transcript])
    except Exception as e:
        return str(e)

# Function to get video data
def get_video_data(video_id):
    video_data = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        id=video_id
    ).execute()
    transcription = get_transcription(video_id)
    video_info = video_data['items'][0]['snippet']
    return {
        'title': video_info['title'],
        'description': video_info['description'],
        'transcription': transcription
    }

# Setup OpenAI's GPT
openai.api_key = openai_api_key

def answer_question(question, video_data):
    context = f"Title: {video_data['title']}\nDescription: {video_data['description']}\nTranscription: {video_data['transcription']}\n"
    print(f"Context for GPT-3: {context}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful teacher."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
            ],
            max_tokens=20
        )
        print(f"GPT-3 Response: {response}")
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."

# Example usage
videos = search_videos('carlsagandotcom pale blue dot')
for video in videos:
    print(f"Title: {video['snippet']['title']}, ID: {video['id']['videoId']}")
    video_data = get_video_data(video['id']['videoId'])
    if video_data:
        question = "What is pale blue dot?"
        answer = answer_question(question, video_data)
        print(f"Answer: {answer}")
    else:
        print(f"Failed to get data for video ID: {video['id']['videoId']}")
