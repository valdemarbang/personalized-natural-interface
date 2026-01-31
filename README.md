# Personalized Natural Interface

A personalized speech-to-text (STT) and text-to-speech (TTS) system with fine-tuning capabilities. The system allows users to create personalized voice profiles by fine-tuning both STT and TTS models on their own voice data.

## Demo

[Watch the demo video](demos/demo_whisperx.webm)

### Samples - Swedish
- [Base model](demos/base_00.wav)
- [Finetuned model](demos/ft_00.wav)

## Architecture

The project consists of several containerized services:

- **Frontend** (Angular): User interface for recording audio, viewing transcripts, and managing profiles
- **Backend** (Flask): API server coordinating between services, managing jobs and user data
- **STT Service** (FastAPI): Speech-to-text transcription and fine-tuning using Whisper
- **TTS Service** (FastAPI): Text-to-speech synthesis and fine-tuning using Chatterbox multilingual TTS
- **LLM Service** (FastAPI): Domain-specific prompt generation using vLLM and Qwen2.5-8B-Instruct

## Installation instruction

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for training)
- NVIDIA Container Toolkit

### Setup

1. Clone the repository
2. Build and start services:

```bash
docker-compose up --build
```

The services will be available at:
- Frontend: http://localhost:4200
- Backend API: http://localhost:5001
- STT Service: http://localhost:5080
- TTS Service: http://localhost:8002
- LLM Service: http://localhost:8001

## Usage

### Domain-Specific Prompt Generation

The LLM service generates tailored training prompts for your specific domain, creating realistic sentences that users can record for STT/TTS training.

#### Generate Prompts

```bash
# Via backend API
POST http://localhost:5001/api/llm/generate-prompts/
{
  "domain": "AI and machine learning",
  "num_prompts": 30,
  "language": "sv",
  "difficulty": "intermediate",
  "sentence_length": "medium",
  "include_technical_terms": true,
  "style": "conversational"
}

# Response:
{
  "prompts": [
    "Neurala nätverk kan lära sig att känna igen mönster i data genom att justera sina vikter.",
    "Transformermodeller använder self-attention för att förstå relationer mellan ord.",
    ...
  ],
  "total": 30
}

# Save generated prompts to domain file
POST http://localhost:5001/api/llm/save-generated-prompts/
{
  "domain_name": "ai_custom",
  "prompts": [
    {"id": 1, "text": "Neurala nätverk..."},
    {"id": 2, "text": "Transformermodeller..."}
  ],
  "language": "sv"
}
```

#### Domain Examples

- **AI/ML**: Technical terms like "neural networks", "backpropagation", "embeddings"
- **Medicine**: Medical terminology, patient interactions, clinical scenarios
- **Finance**: Banking terms, investment discussions, financial reporting
- **Legal**: Legal jargon, case discussions, contract language
- **Customer Service**: Support interactions, troubleshooting, product questions

#### Configuration

The LLM service uses **Qwen2-7B-Instruct** via vLLM for efficient text generation:

```bash
# Load model with custom settings
POST http://localhost:5001/api/llm/load-model/
{
  "model_name": "Qwen/Qwen2-7B-Instruct",
  "tensor_parallel_size": 1,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.9
}

# Check service health
GET http://localhost:5001/api/llm/health/
```

### Fine-tuning Workflow

Both STT and TTS follow a unified training workflow:

#### 1. Prepare Dataset

Record audio samples with corresponding transcripts. The system expects:
- Audio files in WAV format
- For STT: JSONL metadata file with `audio_path` and `text` fields
- For TTS: metadata.txt file with format `filename|transcript`

#### 2. Start Fine-tuning

**STT Fine-tuning:**
```bash
# Via backend API
POST http://localhost:5001/finetuning/start-stt/
{
  "profileID": "user123"
}

# Or directly to STT service
POST http://localhost:5080/load_dataset
{
  "manifest_path": "/app/data/profiles/user123/audio_prompts/metadata.jsonl",
  "recordings_root": "/app/data/profiles/user123/audio_prompts",
  "user": "user123"
}

POST http://localhost:5080/fine_tune
{
  "user": "user123",
  "num_train_epochs": 10,
  "learning_rate": 0.000165
}
```

**TTS Fine-tuning:**
```bash
# Via backend API
POST http://localhost:5001/finetuning/start-tts/
{
  "profileID": "user123"
}

# Or directly to TTS service
POST http://localhost:8000/load_dataset
{
  "manifest_path": "/app/data/profiles/user123/audio_prompts/metadata.txt",
  "recordings_root": "/app/data/profiles/user123/audio_prompts",
  "user": "user123"
}

POST http://localhost:8000/fine_tune
{
  "user": "user123",
  "num_train_epochs": 10,
  "learning_rate": 0.00001
}
```

#### 3. Monitor Progress

```bash
# Check job status via backend
GET http://localhost:5001/finetuning/status/{jobId}

# Or directly check service status
GET http://localhost:5080/job_status/{job_id}  # STT
GET http://localhost:8000/job_status/{job_id}  # TTS
```

### Training Parameters

Common parameters for both STT and TTS:

- `num_train_epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate (STT: 1.65e-4, TTS: 1e-5)
- `per_device_train_batch_size`: Batch size for training (default: 2)
- `gradient_accumulation_steps`: Gradient accumulation (STT: 3, TTS: 4)
- `warmup_steps`: Warmup steps (default: 200)
- `weight_decay`: Weight decay for regularization (default: 0.01)
- `eval_split_size`: Validation split ratio (default: 0.1)

### CLI Usage

Both services also provide CLI interfaces for standalone training:

**STT:**
```bash
cd stt/src/app
python run.py \
  --manifest_path /path/to/metadata.jsonl \
  --recordings_root /path/to/audio \
  --action train
```

**TTS:**
```bash
cd tts/src
python tts_training_service.py \
  --action train \
  --metadata_file /path/to/metadata.txt \
  --dataset_dir /path/to/audio \
  --output_dir ./models/checkpoints/finetuned
```

## Development

### Project Structure

```
├── app/
│   ├── backend/          # Flask API server
│   │   ├── routes/       # API endpoints
│   │   ├── services/     # Business logic (jobs, storage)
│   │   └── data/         # Shared data volume
│   └── frontend/         # Angular UI
├── stt/                  # STT service
│   ├── src/app/         # FastAPI + training code
│   └── Dockerfile
├── tts/                  # TTS service
│   ├── app/             # FastAPI synthesis API
│   ├── src/             # Training service
│   ├── chatterbox/      # Chatterbox TTS scripts
│   └── Dockerfile
├── llm/                  # LLM service
│   ├── app/             # FastAPI + vLLM wrapper
│   └── Dockerfile
└── docker-compose.yml   # Service orchestration
```

### Adding New Training Features

To add features to both services consistently:

1. Update training config dataclasses (STT: `ModelSetup`, TTS: `TTSTrainingConfig`)
2. Add API request models in respective `api_handler.py` / `tts_api.py`
3. Wire to backend routes in `app/backend/routes/finetuning.py`
4. Add job tracking in `app/backend/services/jobs.py`


## License
The project is licensed under the MIT license (https://opensource.org/license/mit).

## Future Work

### Real-time Fine-tuning

The frontend can be enhanced to send 30-second batches of voice data along with their corresponding text output, allowing fine-tuning in near real-time:

- Backend streams audio through Kafka in ~30s chunks (4-6 sentences)
- Real-time streaming through Kafka, continuously fine-tuning models on incoming data
- Eliminates need to wait for complete recording sessions before training
