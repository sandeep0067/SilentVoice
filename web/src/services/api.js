import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add API key if available
const API_KEY = import.meta.env.VITE_API_KEY
if (API_KEY) {
  api.defaults.headers.common['X-API-Key'] = API_KEY
}

export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/api/v1/health')
    return response.data
  },

  // Model info
  async getModelInfo() {
    const response = await api.get('/api/v1/model/info')
    return response.data
  },

  // Single prediction
  async predict(features, options = {}) {
    const response = await api.post('/api/v1/predict', {
      features,
      return_probabilities: options.returnProbabilities || false,
      language: options.language || 'en',
    })
    return response.data
  },

  // Batch prediction
  async predictBatch(sequences, options = {}) {
    const response = await api.post('/api/v1/predict/batch', {
      sequences,
      return_probabilities: options.returnProbabilities || false,
      language: options.language || 'en',
    })
    return response.data
  },

  // Translation
  async translate(classId, confidence = 1.0, language = 'en') {
    const response = await api.post('/api/v1/translate', {
      class_id: classId,
      confidence,
      language,
    })
    return response.data
  },

  // File upload
  async uploadFile(file, autoPredict = true) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('auto_predict', autoPredict)

    const response = await api.post('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Metrics
  async getMetrics() {
    const response = await api.get('/api/v1/metrics')
    return response.data
  },

  // TTS
  async textToSpeech(text, options = {}) {
    const response = await api.post('/api/v1/tts', {
      text,
      language: options.language || 'en',
      voice_id: options.voiceId || 'default',
      speed: options.speed || 1.0,
    })
    return response.data
  },

  // Analytics
  async getPredictionHistory(limit = 100) {
    const response = await api.get('/api/v1/analytics/history', {
      params: { limit },
    })
    return response.data
  },

  async getAnalyticsStatistics() {
    const response = await api.get('/api/v1/analytics/statistics')
    return response.data
  },

  async getPredictionsByClass(classId, limit = 50) {
    const response = await api.get(`/api/v1/analytics/class/${classId}`, {
      params: { limit },
    })
    return response.data
  },

  async getPredictionsByLanguage(language, limit = 50) {
    const response = await api.get(`/api/v1/analytics/language/${language}`, {
      params: { limit },
    })
    return response.data
  },
}

export default apiService
