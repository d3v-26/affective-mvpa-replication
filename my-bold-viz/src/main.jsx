import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './bold-viz.jsx'

createRoot(document.getElementById('root')).render(
  <div style={{ 
      backgroundColor: '#0a0a0a', 
      minHeight: '100vh', 
      width: '100vw',  // Uses full Viewport Width
      margin: 0,
      padding: 0 
    }}>
    <App />
  </div>
)
