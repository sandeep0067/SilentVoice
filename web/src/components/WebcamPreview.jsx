import React, { useRef, useEffect, useState } from 'react'
import { Video, CameraOff, Maximize2, Minimize2 } from 'lucide-react'
import apiService from '../services/api'

function WebcamPreview({ onPrediction, isConnected, setIsConnected, fps, setFps }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [stream, setStream] = useState(null)
  const [useMockData, setUseMockData] = useState(true) // Toggle for testing

  useEffect(() => {
    let animationFrameId
    let lastTime = performance.now()
    let frameCount = 0
    let predictionInterval = null

    const startWebcam = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        })
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
          setStream(mediaStream)
          setIsConnected(true)
        }
      } catch (error) {
        console.error('Error accessing webcam:', error)
        setIsConnected(false)
      }
    }

    const processFrame = () => {
      if (!videoRef.current || !canvasRef.current) return

      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      // Calculate FPS
      frameCount++
      const currentTime = performance.now()
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount)
        frameCount = 0
        lastTime = currentTime
      }

      // Draw video to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      animationFrameId = requestAnimationFrame(processFrame)
    }

    const makePrediction = async () => {
      if (!isConnected) return

      if (useMockData) {
        // Mock prediction for testing
        const mockPrediction = {
          predictedClass: Math.floor(Math.random() * 25),
          predictedLabel: ['Hello', 'Thank you', 'Yes', 'No', 'Please', 'Sorry', 'Good morning', 'How are you'][Math.floor(Math.random() * 8)],
          confidence: 0.7 + Math.random() * 0.3,
          timestamp: new Date().toISOString(),
          inferenceTimeMs: 10 + Math.random() * 20
        }
        onPrediction(mockPrediction)
      } else {
        // Real API prediction (would need MediaPipe integration)
        try {
          // Extract features from canvas
          const canvas = canvasRef.current
          const ctx = canvas.getContext('2d')
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
          
          // Convert to feature array (simplified)
          const features = []
          for (let i = 0; i < 30; i++) {
            features.push(Array(279).fill(0).map(() => Math.random()))
          }
          
          const result = await apiService.predict(features, { returnProbabilities: true })
          
          if (result.success) {
            onPrediction({
              predictedClass: result.predicted_class,
              predictedLabel: result.predicted_label,
              confidence: result.confidence,
              timestamp: new Date().toISOString(),
              inferenceTimeMs: result.inference_time_ms
            })
          }
        } catch (error) {
          console.error('Prediction error:', error)
        }
      }
    }

    startWebcam()
    processFrame()
    
    // Make predictions every 2 seconds
    predictionInterval = setInterval(makePrediction, 2000)

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
      if (predictionInterval) {
        clearInterval(predictionInterval)
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [isConnected, useMockData])

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  return (
    <div className={`glass-card p-6 ${isFullscreen ? 'fixed inset-4 z-50' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <Video className="w-5 h-5 text-primary-400" />
          Camera Feed
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setUseMockData(!useMockData)}
            className={`px-3 py-1 rounded-full text-xs transition-all ${
              useMockData ? 'bg-accent-500/20 text-accent-400' : 'bg-green-500/20 text-green-400'
            }`}
          >
            {useMockData ? 'Demo Mode' : 'Live Mode'}
          </button>
          <div className="text-white/60 text-sm">
            {fps} FPS
          </div>
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all"
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4 text-white" />
            ) : (
              <Maximize2 className="w-4 h-4 text-white" />
            )}
          </button>
        </div>
      </div>

      <div className="relative rounded-xl overflow-hidden bg-black/50">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full aspect-video object-cover"
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="hidden"
        />
        
        {!isConnected && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70">
            <div className="text-center">
              <CameraOff className="w-12 h-12 text-white/40 mx-auto mb-3" />
              <p className="text-white/60">Camera not connected</p>
              <p className="text-white/40 text-sm mt-1">Click to enable camera access</p>
            </div>
          </div>
        )}
        
        {/* Overlay grid */}
        {isConnected && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute inset-0 grid grid-cols-3 grid-rows-3">
              {[...Array(9)].map((_, i) => (
                <div key={i} className="border border-white/10" />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default WebcamPreview
