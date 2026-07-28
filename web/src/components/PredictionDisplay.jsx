import React from 'react'
import { BrainCircuit, Zap, Target } from 'lucide-react'

function PredictionDisplay({ prediction, fps }) {
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceBg = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-500'
    if (confidence >= 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <BrainCircuit className="w-5 h-5 text-accent-400" />
          Live Prediction
        </h2>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/10">
          <Zap className="w-4 h-4 text-yellow-400" />
          <span className="text-white/60 text-sm">{fps} FPS</span>
        </div>
      </div>

      {prediction ? (
        <div className="space-y-4">
          {/* Main Prediction */}
          <div className="bg-gradient-to-r from-primary-500/20 to-accent-500/20 rounded-xl p-6 border border-primary-500/30">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-white/60 text-sm mb-1">Predicted Sign</p>
                <p className="text-3xl font-bold gradient-text">
                  {prediction.predictedLabel}
                </p>
              </div>
              <div className="text-right">
                <p className="text-white/60 text-sm mb-1">Class ID</p>
                <p className="text-2xl font-semibold text-white">
                  #{prediction.predictedClass}
                </p>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-primary-400" />
                  <span className="text-white/80 text-sm">Confidence</span>
                </div>
                <span className={`text-2xl font-bold ${getConfidenceColor(prediction.confidence)}`}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                <div
                  className={`h-full ${getConfidenceBg(prediction.confidence)} transition-all duration-500 ease-out`}
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-white/60 text-xs mb-1">Inference Time</p>
              <p className="text-lg font-semibold text-white">
                {prediction.inferenceTimeMs?.toFixed(2) || '--'} ms
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-white/60 text-xs mb-1">Status</p>
              <p className="text-lg font-semibold text-green-400">
                Active
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-12">
          <BrainCircuit className="w-16 h-16 text-white/20 mx-auto mb-4 animate-pulse-slow" />
          <p className="text-white/40">Waiting for prediction...</p>
          <p className="text-white/30 text-sm mt-2">Make a sign in front of the camera</p>
        </div>
      )}
    </div>
  )
}

export default PredictionDisplay
