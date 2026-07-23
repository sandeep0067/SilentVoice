import { useState } from 'react'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container mx-auto p-8">
        <h1 className="text-4xl font-bold mb-4">SilentVoice</h1>
        <p className="text-muted-foreground mb-8">
          Indian Sign Language Recognition System
        </p>
        <div className="p-6 border rounded-lg">
          <p className="mb-4">Project scaffold created successfully.</p>
          <button
            onClick={() => setCount((c) => c + 1)}
            className="px-4 py-2 bg-primary text-primary-foreground rounded"
          >
            Count: {count}
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
