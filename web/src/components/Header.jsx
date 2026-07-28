import React from 'react'
import { Mic, Wifi, WifiOff, Activity } from 'lucide-react'

function Header({ isConnected }) {
  return (
    <header className="glass-card p-6 mb-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className={`w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center ${isConnected ? 'animate-pulse-slow' : ''}`}>
              <Mic className="w-6 h-6 text-white" />
            </div>
            <div className={`absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-slate-900 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          </div>
          <div>
            <h1 className="text-2xl font-bold gradient-text">SilentVoice</h1>
            <p className="text-white/60 text-sm">Indian Sign Language Recognition</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10">
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-400" />
                <span className="text-green-400 text-sm font-medium">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-400" />
                <span className="text-red-400 text-sm font-medium">Disconnected</span>
              </>
            )}
          </div>
          
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10">
            <Activity className="w-4 h-4 text-primary-400" />
            <span className="text-white/60 text-sm">Real-time</span>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
