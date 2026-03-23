import { useState } from 'react'
import GameDashboard from './components/GameDashboard'
import StrategySummary from './components/StrategySummary'
import Backtester from './components/Backtester'

type Tab = 'dashboard' | 'strategy' | 'backtester'

const API_HOST = `http://${window.location.hostname}:8001`

export { API_HOST }

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard')

  return (
    <div className="app-container">
      <div className="app-header">
        <span className="app-brand">NCAAB</span>
        <button className={`tab-btn ${tab === 'dashboard' ? 'active' : ''}`}
                onClick={() => setTab('dashboard')}>Dashboard</button>
        <button className={`tab-btn ${tab === 'strategy' ? 'active' : ''}`}
                onClick={() => setTab('strategy')}>Strategy</button>
        <button className={`tab-btn ${tab === 'backtester' ? 'active' : ''}`}
                onClick={() => setTab('backtester')}>Backtester</button>
      </div>
      <div className="main-content">
        {tab === 'dashboard' && <GameDashboard />}
        {tab === 'strategy' && <StrategySummary />}
        {tab === 'backtester' && <Backtester />}
      </div>
    </div>
  )
}
