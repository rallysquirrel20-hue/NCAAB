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
  opp_closing_ml: number | null
  covered: boolean | null
  ats_margin: number | null
  team_conference: string
  opp_conference: string
  team_stats: Record<string, number | null>
  opp_stats: Record<string, number | null>
  team_ranks: Record<string, number | null>
  opp_ranks: Record<string, number | null>
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

function formatRawPct(v: number | null): string {
  if (v == null) return '-'
  return `${v.toFixed(1)}%`
}

function formatRank(v: number | null | undefined): string {
  if (v == null) return '-'
  return `#${v}`
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

  // Display order: road team on top for home/away, underdog on top for neutral
  const flip = game.home_away === 'home' ||
    (game.home_away === 'neutral' && game.closing_spread != null && game.closing_spread < 0)

  const top = {
    name: flip ? game.opponent : game.team,
    score: flip ? game.opp_score : game.team_score,
    conf: flip ? game.opp_conference : game.team_conference,
    stats: flip ? game.opp_stats : game.team_stats,
    ranks: flip ? game.opp_ranks : game.team_ranks,
  }
  const bot = {
    name: flip ? game.team : game.opponent,
    score: flip ? game.team_score : game.opp_score,
    conf: flip ? game.team_conference : game.opp_conference,
    stats: flip ? game.team_stats : game.opp_stats,
    ranks: flip ? game.team_ranks : game.opp_ranks,
  }


  // Covered: game.covered is from game.team's perspective
  const isPush = game.ats_margin != null && game.ats_margin === 0
  const teamCovered = game.covered === true && !isPush
  const oppCovered = finished && game.covered === false && game.ats_margin != null && !isPush
  const topCovered = flip ? oppCovered : teamCovered
  const botCovered = flip ? teamCovered : oppCovered
  const topSpread = game.closing_spread != null ? (flip ? -game.closing_spread : game.closing_spread) : null
  const botSpread = game.closing_spread != null ? (flip ? game.closing_spread : -game.closing_spread) : null
  const topMl = flip ? game.opp_closing_ml : game.closing_ml
  const botMl = flip ? game.closing_ml : game.opp_closing_ml

  return (
    <div className="game-card" onClick={() => setExpanded(!expanded)}>
      <div className="game-card-header">
        <span>
          {[
            game.home_away === 'neutral' ? 'Neutral' : null,
            game.conference_game ? 'Conf' : null,
          ].filter(Boolean).join(' | ')}
        </span>
      </div>

      <div className="game-teams">
        <div className="team-row">
          <span className={`team-name ${topCovered ? 'winner' : ''}`}>{top.name}</span>
          <span className="team-line">{formatSpread(topSpread)} ML {formatMl(topMl)}</span>
          <span className="team-score">{top.score ?? '-'}</span>
        </div>
        <div className="team-row">
          <span className={`team-name ${botCovered ? 'winner' : ''}`}>{bot.name}</span>
          <span className="team-line">{formatSpread(botSpread)} ML {formatMl(botMl)}</span>
          <span className="team-score">{bot.score ?? '-'}</span>
        </div>
      </div>

      {expanded && (
        <>
          <div className="game-stats-grid-ranked">
            <div className="stat-cell header">Stat</div>
            <div className="stat-cell header">{top.name.length > 12 ? top.name.slice(0, 12) + '..' : top.name}</div>
            <div className="stat-cell header rank">Rank</div>
            <div className="stat-cell header">{bot.name.length > 12 ? bot.name.slice(0, 12) + '..' : bot.name}</div>
            <div className="stat-cell header rank">Rank</div>

            <div className="stat-cell">Win%</div>
            <div className="stat-cell">{formatPct(top.stats.win_pct)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.win_pct)}</div>
            <div className="stat-cell">{formatPct(bot.stats.win_pct)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.win_pct)}</div>

            <div className="stat-cell">ATS%</div>
            <div className="stat-cell">{formatPct(top.stats.ats_win_pct)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.ats_win_pct)}</div>
            <div className="stat-cell">{formatPct(bot.stats.ats_win_pct)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.ats_win_pct)}</div>

            <div className="stat-cell">PPG</div>
            <div className="stat-cell">{formatStat(top.stats.ppg)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.ppg)}</div>
            <div className="stat-cell">{formatStat(bot.stats.ppg)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.ppg)}</div>

            <div className="stat-cell">3PT%</div>
            <div className="stat-cell">{formatRawPct(top.stats['3pt_pct'])}</div>
            <div className="stat-cell rank">{formatRank(top.ranks['3pt_pct'])}</div>
            <div className="stat-cell">{formatRawPct(bot.stats['3pt_pct'])}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks['3pt_pct'])}</div>

            <div className="stat-cell">FT%</div>
            <div className="stat-cell">{formatRawPct(top.stats.ft_pct)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.ft_pct)}</div>
            <div className="stat-cell">{formatRawPct(bot.stats.ft_pct)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.ft_pct)}</div>

            <div className="stat-cell">Def 2P%</div>
            <div className="stat-cell">{formatRawPct(top.stats.def_2pt_pct)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.def_2pt_pct)}</div>
            <div className="stat-cell">{formatRawPct(bot.stats.def_2pt_pct)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.def_2pt_pct)}</div>

            <div className="stat-cell">OREB/g</div>
            <div className="stat-cell">{formatStat(top.stats.oreb_pg)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.oreb_pg)}</div>
            <div className="stat-cell">{formatStat(bot.stats.oreb_pg)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.oreb_pg)}</div>

            <div className="stat-cell">TO/g</div>
            <div className="stat-cell">{formatStat(top.stats.to_pg)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.to_pg)}</div>
            <div className="stat-cell">{formatStat(bot.stats.to_pg)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.to_pg)}</div>

            <div className="stat-cell">Pace</div>
            <div className="stat-cell">{formatStat(top.stats.pace)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.pace)}</div>
            <div className="stat-cell">{formatStat(bot.stats.pace)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.pace)}</div>

            <div className="stat-cell">SOS</div>
            <div className="stat-cell">{formatStat(top.stats.sos)}</div>
            <div className="stat-cell rank">{formatRank(top.ranks.sos)}</div>
            <div className="stat-cell">{formatStat(bot.stats.sos)}</div>
            <div className="stat-cell rank">{formatRank(bot.ranks.sos)}</div>
          </div>

          <LineMovementChart eventId={game.event_id} />
        </>
      )}
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

      {/* Game data cards */}
      {games.length > 0 && (
        <div className="games-grid">
          {games.map(g => <GameCard key={`${g.event_id}_${g.team}`} game={g} />)}
        </div>
      )}

      {!loading && games.length === 0 && (
        <div className="empty-state">No games found for {selectedDate}</div>
      )}
    </>
  )
}
