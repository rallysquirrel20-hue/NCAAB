import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import { createChart, type IChartApi } from 'lightweight-charts'
import { API_HOST } from '../App'

interface GameData {
  event_id: string
  team: string
  opponent: string
  home_away: string
  neutral_site: boolean
  conference_game: boolean
  team_score: number | null
  opp_score: number | null
  win_loss: string
  opening_spread: number | null
  closing_spread: number | null
  opening_ml: number | null
  closing_ml: number | null
  covered: boolean | null
  ats_margin: number | null
  team_conference: string
  opp_conference: string
  team_stats: Record<string, number | null>
  opp_stats: Record<string, number | null>
}

interface ESPNGame {
  event_id: string
  state: string
  detail: string
  clock: string
  period: number
  home: { name: string; short: string; abbreviation: string; score: number; seed: number }
  away: { name: string; short: string; abbreviation: string; score: number; seed: number }
  neutral_site: boolean
  start_time: string
}

interface LineMovement {
  time: string
  bookmaker: string
  outcome: string
  point: number | null
  price: number | null
}

function formatSpread(v: number | null): string {
  if (v == null) return '-'
  return v > 0 ? `+${v}` : `${v}`
}

function formatMl(v: number | null): string {
  if (v == null) return '-'
  return v > 0 ? `+${v}` : `${v}`
}

function formatPct(v: number | null): string {
  if (v == null) return '-'
  return `${(v * 100).toFixed(1)}%`
}

function formatStat(v: number | null, decimals = 1): string {
  if (v == null) return '-'
  return v.toFixed(decimals)
}

function todayStr(): string {
  const d = new Date()
  return d.toISOString().slice(0, 10)
}

function shiftDate(dateStr: string, days: number): string {
  const d = new Date(dateStr + 'T12:00:00')
  d.setDate(d.getDate() + days)
  return d.toISOString().slice(0, 10)
}

// ── Line movement mini chart ──────────────────────────────────────────────
function LineMovementChart({ eventId }: { eventId: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [movements, setMovements] = useState<LineMovement[]>([])

  useEffect(() => {
    if (!eventId) return
    axios.get(`${API_HOST}/api/games/${eventId}/line-movement`)
      .then(r => setMovements(r.data.movements || []))
      .catch(() => setMovements([]))
  }, [eventId])

  useEffect(() => {
    if (!containerRef.current || movements.length === 0) return

    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 100,
      layout: { background: { color: '#eee8d5' }, textColor: '#586e75', fontSize: 9 },
      grid: { vertLines: { visible: false }, horzLines: { color: '#93a1a1', style: 2 } },
      timeScale: { visible: false },
      rightPriceScale: { borderVisible: false },
      crosshair: { mode: 0 },
    })
    chartRef.current = chart

    // Group by outcome (home/away spread lines)
    const outcomes = new Map<string, { time: string; value: number }[]>()
    for (const m of movements) {
      if (m.point == null) continue
      const key = m.outcome || 'spread'
      if (!outcomes.has(key)) outcomes.set(key, [])
      outcomes.get(key)!.push({
        time: m.time.slice(0, 10),
        value: m.point,
      })
    }

    const colors = ['rgb(50, 50, 255)', 'rgb(255, 50, 150)']
    let colorIdx = 0
    for (const [, points] of outcomes) {
      // Deduplicate by date (keep last value per date)
      const byDate = new Map<string, number>()
      for (const p of points) byDate.set(p.time, p.value)
      const data = Array.from(byDate.entries())
        .map(([time, value]) => ({ time, value }))
        .sort((a, b) => a.time.localeCompare(b.time))

      if (data.length > 0) {
        const series = chart.addLineSeries({
          color: colors[colorIdx % colors.length],
          lineWidth: 2,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        })
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        series.setData(data as any)
        colorIdx++
      }
    }

    chart.timeScale().fitContent()

    return () => {
      chart.remove()
      chartRef.current = null
    }
  }, [movements])

  if (movements.length === 0) return null

  return (
    <div className="line-movement-container">
      <div className="line-movement-label">Line Movement</div>
      <div ref={containerRef} style={{ width: '100%', height: 100 }} />
    </div>
  )
}


// ── Game card ─────────────────────────────────────────────────────────────
function GameCard({ game }: { game: GameData }) {
  const [expanded, setExpanded] = useState(false)
  const finished = game.team_score != null && game.opp_score != null
  const teamWon = finished && game.win_loss === 'W'

  return (
    <div className="game-card" onClick={() => setExpanded(!expanded)}>
      <div className="game-card-header">
        <span>
          {game.home_away === 'neutral' ? 'Neutral' :
           game.home_away === 'home' ? 'Home' : 'Away'}
          {game.conference_game ? ' | Conf' : ''}
        </span>
        {finished && game.covered !== null && (
          <span className={game.covered ? 'ats-cover' : 'ats-miss'}>
            {game.covered ? 'COVERED' : 'MISSED'} ({game.ats_margin != null ? (game.ats_margin > 0 ? '+' : '') + game.ats_margin.toFixed(1) : ''})
          </span>
        )}
      </div>

      <div className="game-teams">
        <div className="team-row">
          <span className={`team-name ${teamWon ? 'winner' : ''}`}>{game.team}</span>
          <span className="team-record">{game.team_conference}</span>
          <span className="team-score">{game.team_score ?? '-'}</span>
        </div>
        <div className="team-row">
          <span className="team-name">{game.opponent}</span>
          <span className="team-record">{game.opp_conference}</span>
          <span className="team-score">{game.opp_score ?? '-'}</span>
        </div>
      </div>

      <div className="game-spread-row">
        <span><span className="spread-label">Open: </span>{formatSpread(game.opening_spread)}</span>
        <span><span className="spread-label">Close: </span>{formatSpread(game.closing_spread)}</span>
        <span><span className="spread-label">ML: </span>{formatMl(game.closing_ml)}</span>
      </div>

      {expanded && (
        <>
          <div className="game-stats-grid">
            <div className="stat-cell header">Stat</div>
            <div className="stat-cell header">{game.team.length > 12 ? game.team.slice(0, 12) + '..' : game.team}</div>
            <div className="stat-cell header">{(game.opponent || '').length > 12 ? game.opponent.slice(0, 12) + '..' : game.opponent}</div>
            <div className="stat-cell header">Edge</div>

            <div className="stat-cell">Win%</div>
            <div className="stat-cell">{formatPct(game.team_stats.win_pct)}</div>
            <div className="stat-cell">{formatPct(game.opp_stats.win_pct)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.win_pct} b={game.opp_stats.win_pct} isPct={true} /></div>

            <div className="stat-cell">ATS%</div>
            <div className="stat-cell">{formatPct(game.team_stats.ats_win_pct)}</div>
            <div className="stat-cell">{formatPct(game.opp_stats.ats_win_pct)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.ats_win_pct} b={game.opp_stats.ats_win_pct} isPct={true} /></div>

            <div className="stat-cell">PPG</div>
            <div className="stat-cell">{formatStat(game.team_stats.ppg)}</div>
            <div className="stat-cell">{formatStat(game.opp_stats.ppg)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.ppg} b={game.opp_stats.ppg} isPct={false} /></div>

            <div className="stat-cell">3PT%</div>
            <div className="stat-cell">{formatPct(game.team_stats['3pt_pct'])}</div>
            <div className="stat-cell">{formatPct(game.opp_stats['3pt_pct'])}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats['3pt_pct']} b={game.opp_stats['3pt_pct']} isPct={true} /></div>

            <div className="stat-cell">FT%</div>
            <div className="stat-cell">{formatPct(game.team_stats.ft_pct)}</div>
            <div className="stat-cell">{formatPct(game.opp_stats.ft_pct)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.ft_pct} b={game.opp_stats.ft_pct} isPct={true} /></div>

            <div className="stat-cell">Def 2P%</div>
            <div className="stat-cell">{formatPct(game.team_stats.def_2pt_pct)}</div>
            <div className="stat-cell">{formatPct(game.opp_stats.def_2pt_pct)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.def_2pt_pct} b={game.opp_stats.def_2pt_pct} isPct={false} /></div>

            <div className="stat-cell">OREB/g</div>
            <div className="stat-cell">{formatStat(game.team_stats.oreb_pg)}</div>
            <div className="stat-cell">{formatStat(game.opp_stats.oreb_pg)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.oreb_pg} b={game.opp_stats.oreb_pg} isPct={false} /></div>

            <div className="stat-cell">TO/g</div>
            <div className="stat-cell">{formatStat(game.team_stats.to_pg)}</div>
            <div className="stat-cell">{formatStat(game.opp_stats.to_pg)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.to_pg} b={game.opp_stats.to_pg} isPct={false} lowerBetter={true} /></div>

            <div className="stat-cell">Pace</div>
            <div className="stat-cell">{formatStat(game.team_stats.pace)}</div>
            <div className="stat-cell">{formatStat(game.opp_stats.pace)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.pace} b={game.opp_stats.pace} isPct={false} /></div>

            <div className="stat-cell">SOS</div>
            <div className="stat-cell">{formatStat(game.team_stats.sos)}</div>
            <div className="stat-cell">{formatStat(game.opp_stats.sos)}</div>
            <div className="stat-cell"><EdgeCell a={game.team_stats.sos} b={game.opp_stats.sos} isPct={false} lowerBetter={true} /></div>
          </div>

          <LineMovementChart eventId={game.event_id} />
        </>
      )}
    </div>
  )
}

function EdgeCell({ a, b, isPct, lowerBetter = false }: { a: number | null | undefined; b: number | null | undefined; isPct: boolean; lowerBetter?: boolean }) {
  if (a == null || b == null) return <>-</>
  const diff = a - b
  const adjusted = lowerBetter ? -diff : diff
  const formatted = isPct ? `${(diff * 100).toFixed(1)}%` : diff.toFixed(1)
  const sign = diff > 0 ? '+' : ''
  const cls = adjusted > 0 ? 'positive' : adjusted < 0 ? 'negative' : ''
  return <span className={cls}>{sign}{formatted}</span>
}


// ── Live game card ────────────────────────────────────────────────────────
function LiveGameCard({ game }: { game: ESPNGame }) {
  const isLive = game.state === 'in'

  return (
    <div className={`game-card ${isLive ? 'live' : ''}`}>
      <div className="game-card-header">
        <span>
          {isLive && <span className="live-dot" />}
          {game.detail}
        </span>
        <span>{game.neutral_site ? 'Neutral' : ''}</span>
      </div>
      <div className="game-teams">
        <div className="team-row">
          {game.away.seed > 0 && <span className="team-seed">({game.away.seed})</span>}
          <span className="team-name">{game.away.name}</span>
          <span className="team-score">{game.state !== 'pre' ? game.away.score : '-'}</span>
        </div>
        <div className="team-row">
          {game.home.seed > 0 && <span className="team-seed">({game.home.seed})</span>}
          <span className="team-name">{game.home.name}</span>
          <span className="team-score">{game.state !== 'pre' ? game.home.score : '-'}</span>
        </div>
      </div>
    </div>
  )
}


// ── Main Dashboard ────────────────────────────────────────────────────────
export default function GameDashboard() {
  const [selectedDate, setSelectedDate] = useState(todayStr())
  const [games, setGames] = useState<GameData[]>([])
  const [liveGames, setLiveGames] = useState<ESPNGame[]>([])
  const [loading, setLoading] = useState(false)
  const [showLive, setShowLive] = useState(true)
  const liveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchGames = useCallback((date: string) => {
    setLoading(true)
    axios.get(`${API_HOST}/api/games`, { params: { date } })
      .then(r => setGames(r.data.games || []))
      .catch(() => setGames([]))
      .finally(() => setLoading(false))
  }, [])

  const fetchLive = useCallback(() => {
    const espnDate = selectedDate.replace(/-/g, '')
    axios.get(`${API_HOST}/api/espn/scoreboard`, { params: { date: espnDate } })
      .then(r => setLiveGames(r.data.games || []))
      .catch(() => setLiveGames([]))
  }, [selectedDate])

  useEffect(() => {
    fetchGames(selectedDate)
    fetchLive()
  }, [selectedDate, fetchGames, fetchLive])

  // Poll ESPN every 30s when showing live games
  useEffect(() => {
    if (showLive) {
      liveIntervalRef.current = setInterval(fetchLive, 30000)
    }
    return () => {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current)
    }
  }, [showLive, fetchLive])

  const hasLive = liveGames.some(g => g.state === 'in')

  return (
    <>
      <div className="date-picker-row">
        <button className="date-btn" onClick={() => setSelectedDate(shiftDate(selectedDate, -1))}>&#9664;</button>
        <input type="date" className="date-input" value={selectedDate}
               onChange={e => setSelectedDate(e.target.value)} />
        <button className="date-btn" onClick={() => setSelectedDate(shiftDate(selectedDate, 1))}>&#9654;</button>
        <button className="date-btn" onClick={() => setSelectedDate(todayStr())}>Today</button>
        <button className={`tab-btn ${showLive ? 'active' : ''}`}
                onClick={() => setShowLive(!showLive)}>
          {hasLive && <span className="live-dot" />}
          ESPN Live
        </button>
        <span style={{ fontSize: 11, color: 'var(--base00)' }}>
          {games.length} game{games.length !== 1 ? 's' : ''} from data
          {showLive && ` | ${liveGames.length} from ESPN`}
        </span>
      </div>

      {loading && <div className="loading">Loading games...</div>}

      {/* Live ESPN games */}
      {showLive && liveGames.length > 0 && (
        <>
          <div style={{ fontSize: 12, fontWeight: 'bold', color: 'var(--text-bold)', marginBottom: 8 }}>
            ESPN Scoreboard ({selectedDate})
          </div>
          <div className="games-grid" style={{ marginBottom: 16 }}>
            {liveGames.map(g => <LiveGameCard key={g.event_id} game={g} />)}
          </div>
        </>
      )}

      {/* Historical/data games */}
      {games.length > 0 && (
        <>
          <div style={{ fontSize: 12, fontWeight: 'bold', color: 'var(--text-bold)', marginBottom: 8 }}>
            Game Data ({selectedDate}) — click to expand stats
          </div>
          <div className="games-grid">
            {games.map(g => <GameCard key={`${g.event_id}_${g.team}`} game={g} />)}
          </div>
        </>
      )}

      {!loading && games.length === 0 && liveGames.length === 0 && (
        <div className="empty-state">No games found for {selectedDate}</div>
      )}
    </>
  )
}
