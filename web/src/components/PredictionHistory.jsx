import React from 'react'
import { Clock, TrendingUp, Trash2 } from 'lucide-react'

function PredictionHistory({ history, onSelectPrediction }) {
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-500'
    if (confidence >= 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const formatTime = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const clearHistory = () => {
    // In production, this would clear the history
    console.log('Clear history')
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <Clock className="w-5 h-5 text-primary-400" />
          Recent Predictions
        </h2>
        <button
          onClick={clearHistory}
          className="p-2 rounded-lg bg-white/10 hover:bg-red-500/20 transition-all group"
          title="Clear history"
        >
          <Trash2 className="w-4 h-4 text-white/60 group-hover:text-red-400 transition-colors" />
        </button>
      </div>

      {history.length === 0 ? (
        <div className="text-center py-8">
          <Clock className="w-12 h-12 text-white/20 mx-auto mb-3" />
          <p className="text-white/40 text-sm">No predictions yet</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar">
          {history.map((prediction, index) => (
            <div
              key={index}
              onClick={() => onSelectPrediction(prediction)}
              className="bg-white/5 rounded-lg p-4 hover:bg-white/10 transition-all cursor-pointer group border border-transparent hover:border-white/20"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg font-semibold text-white">
                    {prediction.predictedLabel}
                  </span>
                  <span className="text-xs text-white/40">#{prediction.predictedClass}</span>
                </div>
                <span className="text-xs text-white/40">
                  {formatTime(prediction.timestamp)}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-24 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${getConfidenceColor(prediction.confidence)}`}
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-white/60">
                    {(prediction.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <TrendingUp className="w-4 h-4 text-white/20 group-hover:text-primary-400 transition-colors" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats Summary */}
      {history.length > 0 && (
        <div className="mt-4 pt-4 border-t border-white/10">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-white">{history.length}</p>
              <p className="text-xs text-white/40">Total</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-400">
                {history.filter(p => p.confidence >= 0.8).length}
              </p>
              <p className="text-xs text-white/40">High Confidence</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default PredictionHistory
