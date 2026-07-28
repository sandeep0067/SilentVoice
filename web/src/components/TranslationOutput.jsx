import React, { useState } from 'react'
import { Volume2, VolumeX, Languages, Copy, Check } from 'lucide-react'
import apiService from '../services/api'

function TranslationOutput({ prediction, onSpeak, isSpeaking }) {
  const [copied, setCopied] = useState(false)
  const [selectedLanguage, setSelectedLanguage] = useState('en')
  const [isTTSLoading, setIsTTSLoading] = useState(false)

  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'hi', name: 'Hindi', flag: '🇮🇳' },
    { code: 'bn', name: 'Bengali', flag: '🇧🇩' },
    { code: 'ta', name: 'Tamil', flag: '🇮🇳' },
    { code: 'te', name: 'Telugu', flag: '🇮🇳' },
  ]

  const handleCopy = () => {
    if (prediction?.predictedLabel) {
      navigator.clipboard.writeText(prediction.predictedLabel)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleSpeak = async () => {
    if (prediction?.predictedLabel) {
      setIsTTSLoading(true)
      try {
        // Try API TTS first
        const result = await apiService.textToSpeech(prediction.predictedLabel, {
          language: selectedLanguage
        })
        
        if (result.success && result.audio_data) {
          // Play audio from API response
          const audioBlob = new Blob([result.audio_data], { type: 'audio/wav' })
          const audioUrl = URL.createObjectURL(audioBlob)
          const audio = new Audio(audioUrl)
          audio.play()
          onSpeak()
        } else {
          // Fallback to browser TTS
          const utterance = new SpeechSynthesisUtterance(prediction.predictedLabel)
          utterance.lang = selectedLanguage === 'hi' ? 'hi-IN' : 'en-US'
          speechSynthesis.speak(utterance)
          onSpeak()
        }
      } catch (error) {
        console.error('TTS error:', error)
        // Fallback to browser TTS
        const utterance = new SpeechSynthesisUtterance(prediction.predictedLabel)
        utterance.lang = selectedLanguage === 'hi' ? 'hi-IN' : 'en-US'
        speechSynthesis.speak(utterance)
        onSpeak()
      } finally {
        setIsTTSLoading(false)
      }
    }
  }

  const getTranslation = (lang) => {
    // In production, this would call the translation API
    const translations = {
      'en': prediction?.predictedLabel || '',
      'hi': 'नमस्ते', // Example - would be from API
      'bn': 'হ্যালো',
      'ta': 'வணக்கம்',
      'te': 'హలో',
    }
    return translations[lang] || prediction?.predictedLabel || ''
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <Languages className="w-5 h-5 text-primary-400" />
          Translation
        </h2>
        <div className="flex gap-2">
          {languages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => setSelectedLanguage(lang.code)}
              className={`px-3 py-1 rounded-full text-sm transition-all ${
                selectedLanguage === lang.code
                  ? 'bg-primary-500 text-white'
                  : 'bg-white/10 text-white/60 hover:bg-white/20'
              }`}
            >
              {lang.flag} {lang.name}
            </button>
          ))}
        </div>
      </div>

      {prediction ? (
        <div className="space-y-4">
          {/* Translation Output */}
          <div className="bg-gradient-to-br from-accent-500/20 to-primary-500/20 rounded-xl p-6 border border-accent-500/30">
            <p className="text-white/60 text-sm mb-2">Translated Text</p>
            <p className="text-2xl font-bold text-white mb-4 animate-float">
              {getTranslation(selectedLanguage)}
            </p>
            
            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={handleSpeak}
                disabled={isSpeaking || isTTSLoading}
                className={`glow-button flex-1 py-3 px-4 rounded-xl bg-white/10 hover:bg-white/20 transition-all flex items-center justify-center gap-2 ${
                  isSpeaking || isTTSLoading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isTTSLoading ? (
                  <>
                    <Volume2 className="w-5 h-5 animate-pulse" />
                    <span>Loading...</span>
                  </>
                ) : isSpeaking ? (
                  <>
                    <VolumeX className="w-5 h-5" />
                    <span>Speaking...</span>
                  </>
                ) : (
                  <>
                    <Volume2 className="w-5 h-5" />
                    <span>Speak</span>
                  </>
                )}
              </button>
              
              <button
                onClick={handleCopy}
                className="py-3 px-4 rounded-xl bg-white/10 hover:bg-white/20 transition-all flex items-center justify-center gap-2"
              >
                {copied ? (
                  <>
                    <Check className="w-5 h-5 text-green-400" />
                    <span className="text-green-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-5 h-5" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Language Info */}
          <div className="bg-white/5 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white/60 text-xs">Detected Language</p>
                <p className="text-white font-medium">English</p>
              </div>
              <div>
                <p className="text-white/60 text-xs">Target Language</p>
                <p className="text-white font-medium">
                  {languages.find(l => l.code === selectedLanguage)?.name}
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-12">
          <Languages className="w-16 h-16 text-white/20 mx-auto mb-4 animate-pulse-slow" />
          <p className="text-white/40">No translation yet</p>
          <p className="text-white/30 text-sm mt-2">Make a sign to see translation</p>
        </div>
      )}
    </div>
  )
}

export default TranslationOutput
