# SilentVoice Web UI

Premium React-based user interface for Indian Sign Language Recognition.

## Features

- **Real-time Webcam Preview** - Live camera feed with FPS monitoring
- **Live Prediction Display** - Shows predicted signs with confidence indicators
- **Multi-language Translation** - Support for English, Hindi, Bengali, Tamil, Telugu
- **Text-to-Speech** - Speak translations with API or browser TTS fallback
- **Prediction History** - Track recent predictions with analytics
- **Premium UI** - Glass morphism design with smooth animations

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first CSS framework
- **Lucide React** - Icon library
- **Axios** - HTTP client for API integration

## Installation

```bash
cd web
npm install
```

## Configuration

Create a `.env` file based on `.env.example`:

```env
VITE_API_URL=http://localhost:8000
VITE_API_KEY=dev-key-12345
```

## Development

```bash
npm run dev
```

The UI will be available at `http://localhost:3000`

## Build

```bash
npm run build
```

## Components

- `App.jsx` - Main application component
- `Header.jsx` - Header with connection status
- `WebcamPreview.jsx` - Camera feed with prediction integration
- `PredictionDisplay.jsx` - Live prediction with confidence indicator
- `TranslationOutput.jsx` - Multi-language translation with TTS
- `PredictionHistory.jsx` - Recent predictions panel

## API Integration

The UI connects to the FastAPI backend via the `apiService` in `src/services/api.js`.

## Features

### Demo Mode
The webcam component includes a demo mode toggle for testing without a trained model:
- **Demo Mode** - Generates mock predictions for UI testing
- **Live Mode** - Connects to the actual API for real predictions

### Language Support
- English (🇬🇧)
- Hindi (🇮🇳)
- Bengali (🇧🇩)
- Tamil (🇮🇳)
- Telugu (🇮🇳)

### TTS Integration
- Primary: API-based TTS (if enabled)
- Fallback: Browser SpeechSynthesis API
