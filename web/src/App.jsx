import React, { useState, useEffect } from 'react'
import WebcamPreview from './components/WebcamPreview'
import PredictionDisplay from './components/PredictionDisplay'
import TranslationOutput from './components/TranslationOutput'
import PredictionHistory from './components/PredictionHistory'
import Header from './components/Header'
import { Mic, Volume2, History, Settings, Wifi, WifiOff } from 'lucide-react'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [currentPrediction, setCurrentPrediction] = useState(null)
  const [history, setHistory] = useState([])
  const [fps, setFps] = useState(0)

  const handlePrediction = (prediction) => {
    setCurrentPrediction(prediction)
    setHistory(prev => [prediction, ...prev].slice(0, 50))
  }

  const handleSpeak = () => {
    if (currentPrediction && currentPrediction.translatedPhrase) {
      setIsSpeaking(true)
      // TTS integration would go here
      setTimeout(() => setIsSpeaking(false), 2000)
    }
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      <Header isConnected={isConnected} />
      
      <div className="max-w-7xl mx-auto mt-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Webcam */}
          <div className="lg:col-span-2 space-y-6">
            <WebcamPreview 
              onPrediction={handlePrediction}
              isConnected={isConnected}
              setIsConnected={setIsConnected}
              fps={fps}
              setFps={setFps}
            />
            
            {/* Prediction Display */}
            <PredictionDisplay 
              prediction={currentPrediction}
              fps={fps}
            />
          </div>
          
          {/* Right Column - Translation & History */}
          <div className="space-y-6">
            {/* Translation Output */}
            <TranslationOutput 
              prediction={currentPrediction}
              onSpeak={handleSpeak}
              isSpeaking={isSpeaking}
            />
            
            {/* Prediction History Toggle */}
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="w-full glass-card p-4 flex items-center justify-between hover:bg-white/20 transition-all duration-300 group"
            >
              <div className="flex items-center gap-3">
                <History className="w-5 h-5 text-primary-400 group-hover:text-primary-300 transition-colors" />
                <span className="text-white font-medium">Prediction History</span>
              </div>
              <div className="text-white/60 text-sm">
                {history.length} predictions
              </div>
            </button>
            
            {/* Prediction History Panel */}
            {showHistory && (
              <PredictionHistory 
                history={history}
                onSelectPrediction={setCurrentPrediction}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
