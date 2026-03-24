import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import { createChart, LineSeries } from 'lightweight-charts'
import type { IChartApi } from 'lightweight-charts'
import './index.css'

const API_BASE = `http://${window.location.hostname}:8001`

// ============================================================================
// Types
// ============================================================================

interface TeamSide {
  name: string
  display_name: string
  abbreviation: string
  id: string
  score: number | null
  is_tracked: boolean
  stats?: Record<string, any>
}

interface GameOdds {
  spread: number | null
  spread_price: number | null
  away_spread: number | null
  away_spread_price: number | null
  ml: number | null
  away_ml: number | null
  total: number | null
  over_price: number | null
  under_price: number | null
  bookmaker?: string
  // Legacy fields from today endpoint
  opening_spread?: number | null
  opening_ml?: number | null
  away_opening_spread?: number | null
  away_opening_ml?: number | null
  bookmakers?: Array<{
    name: string
    markets: Record<string, any>
  }>
}

interface Game {
  game_id: string
  commence_time: string
  status: string
  status_detail: string
  neutral_site: boolean
  home: TeamSide
  away: TeamSide
  odds: GameOdds | null
}

interface TodayData {
  updated_at: string | null
  date: string
  game_count: number
  games: Game[]
}

interface BacktestResult {
  name: string
  n: number
  wins?: number
  losses?: number
  win_pct?: number
  roi?: number
  p_value?: number
  ci_low?: number
  ci_high?: number
  profitable?: boolean
}

interface BacktestGame {
  date: string
  team: string
  opponent: string
  home_away: string
  spread: number | null
  team_score: number
  opp_score: number
  margin: number
  ats_margin: number
  covered: boolean
}

interface PnlPoint {
  date: string
  pnl: number
}

interface Filter {
  stat: string
  op: string
  value: string
  compare_col?: string  // column-vs-column mode
}

interface ColEntry {
  col: string
  label: string
}

interface ColumnGroups {
  [group: string]: ColEntry[]
}

type View = 'today' | 'backtest'

// ============================================================================
// Helpers
// ============================================================================

function formatTime(isoStr: string | undefined, dateStr?: string): string {
  if (!isoStr && dateStr) {
    // Historical game — show the date
    try {
      const d = new Date(dateStr + 'T12:00:00')
      return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
    } catch { return dateStr }
  }
  if (!isoStr) return ''
  try {
    const d = new Date(isoStr)
    if (isNaN(d.getTime())) {
      // Fallback: try treating as date string
      return dateStr || isoStr
    }
    // If time is midnight (placeholder), show date instead
    if (d.getUTCHours() === 0 && d.getUTCMinutes() === 0) {
      return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
    }
    return d.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
      timeZoneName: 'short',
    })
  } catch {
    return dateStr || isoStr
  }
}

function formatSpread(val: number | null | undefined): string {
  if (val == null) return '-'
  return val > 0 ? `+${val}` : `${val}`
}

function formatML(val: number | null | undefined): string {
  if (val == null) return '-'
  return val > 0 ? `+${val}` : `${val}`
}

function getStatusClass(status: string): string {
  if (status === 'STATUS_IN_PROGRESS' || status === 'STATUS_HALFTIME' || status === 'STATUS_END_PERIOD') return 'live'
  if (status === 'STATUS_FINAL') return 'final'
  return 'scheduled'
}

function getStatusLabel(status: string, detail: string): string {
  if (status === 'STATUS_FINAL') return 'FINAL'
  if (status === 'STATUS_IN_PROGRESS') return detail || 'LIVE'
  if (status === 'STATUS_HALFTIME') return 'HALF'
  if (status === 'STATUS_SCHEDULED') return 'SCHEDULED'
  return status.replace('STATUS_', '')
}

// ============================================================================
// MatchupTable Component
// ============================================================================

const MATCHUP_STATS: Array<{ key: string; label: string; higherBetter: boolean }> = [
  { key: 'team_win_pct', label: 'Win %', higherBetter: true },
  { key: 'team_ats_win_pct', label: 'ATS %', higherBetter: true },
  { key: 'team_ppg', label: 'Off PPG', higherBetter: true },
  { key: 'opp_ppg', label: 'Def PPG', higherBetter: false },
  { key: 'team_3pt_pct', label: '3PT %', higherBetter: true },
  { key: 'team_2pt_pct', label: '2PT %', higherBetter: true },
  { key: 'team_ft_pct', label: 'FT %', higherBetter: true },
  { key: 'team_def_3pt_pct', label: 'Def 3PT %', higherBetter: false },
  { key: 'team_def_2pt_pct', label: 'Def 2PT %', higherBetter: false },
  { key: 'team_oreb_pg', label: 'OREB/G', higherBetter: true },
  { key: 'team_dreb_pg', label: 'DREB/G', higherBetter: true },
  { key: 'team_to_pg', label: 'TO/G', higherBetter: false },
  { key: 'team_forced_to_pg', label: 'Forced TO/G', higherBetter: true },
  { key: 'team_pace', label: 'Pace', higherBetter: true },
  { key: 'team_sos', label: 'SOS', higherBetter: false },
]

// Default pairing: team offensive stat → opponent's corresponding defensive stat
const STAT_VS_DEFAULT: Record<string, string> = {
  'team_3pt_pct': 'opp_def_3pt_pct',
  'team_2pt_pct': 'opp_def_2pt_pct',
  'team_ft_pct': 'opp_def_ft_pct',
  'team_def_3pt_pct': 'opp_3pt_pct',
  'team_def_2pt_pct': 'opp_2pt_pct',
  'team_ppg': 'opp_ppg',
  'team_win_pct': 'opp_win_pct',
  'team_ats_win_pct': 'opp_ats_win_pct',
  'team_pace': 'opp_pace',
  'team_oreb_pg': 'opp_oreb_pg',
  'team_dreb_pg': 'opp_dreb_pg',
  'team_to_pg': 'opp_forced_to_pg',
  'team_forced_to_pg': 'opp_to_pg',
  'team_sos': 'opp_sos',
}

// All opponent columns available for vs-comparison
const OPP_COLUMNS = [
  { group: 'Record', cols: [
    { col: 'opp_win_pct', label: 'Opp Win %' },
    { col: 'opp_ats_win_pct', label: 'Opp ATS %' },
  ]},
  { group: 'Scoring', cols: [
    { col: 'opp_ppg', label: 'Opp Off PPG' },
    { col: 'opp_home_ppg', label: 'Opp Home PPG' },
    { col: 'opp_away_ppg', label: 'Opp Away PPG' },
  ]},
  { group: 'Shooting', cols: [
    { col: 'opp_ft_pct', label: 'Opp FT %' },
    { col: 'opp_3pt_pct', label: 'Opp 3PT %' },
    { col: 'opp_2pt_pct', label: 'Opp 2PT %' },
  ]},
  { group: 'Defense', cols: [
    { col: 'opp_def_ft_pct', label: 'Opp Def FT %' },
    { col: 'opp_def_3pt_pct', label: 'Opp Def 3PT %' },
    { col: 'opp_def_2pt_pct', label: 'Opp Def 2PT %' },
  ]},
  { group: 'Rebounding', cols: [
    { col: 'opp_oreb_pg', label: 'Opp OREB/G' },
    { col: 'opp_dreb_pg', label: 'Opp DREB/G' },
  ]},
  { group: 'Turnovers', cols: [
    { col: 'opp_to_pg', label: 'Opp TO/G' },
    { col: 'opp_forced_to_pg', label: 'Opp Forced TO/G' },
  ]},
  { group: 'Pace/SOS', cols: [
    { col: 'opp_pace', label: 'Opp Pace' },
    { col: 'opp_sos', label: 'Opp SOS' },
  ]},
]

// Keep old name as alias for the click handler
const STAT_VS_PAIRS = STAT_VS_DEFAULT

function VsCompareSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <select className="vs-select" value={value} onChange={e => onChange(e.target.value)}>
      {OPP_COLUMNS.map(({ group, cols }) => (
        <optgroup key={group} label={group}>
          {cols.map(c => <option key={c.col} value={c.col}>{c.label}</option>)}
        </optgroup>
      ))}
    </select>
  )
}

function MatchupTable({ gameId, onStatClick }: { gameId: string; onStatClick?: (stat: string, op: string, value: string, label: string, compare_col?: string) => void }) {
  const [matchup, setMatchup] = useState<any>(null)

  useEffect(() => {
    axios.get(`${API_BASE}/api/game/${gameId}/matchup`)
      .then(r => setMatchup(r.data))
      .catch(() => {})
  }, [gameId])

  if (!matchup) return <div className="loading">Loading matchup...</div>

  const home = matchup.home || {}
  const away = matchup.away || {}
  const homeRanks: Record<string, number> = home.ranks || {}
  const awayRanks: Record<string, number> = away.ranks || {}

  const handleStatClick = (key: string, value: any, higherBetter: boolean, label: string) => {
    if (value == null || !onStatClick) return
    const vsPair = STAT_VS_PAIRS[key]
    if (vsPair) {
      // Column-vs-column: e.g., team_3pt_pct > opp_def_3pt_pct by 0+
      const op = higherBetter ? '>' : '<'
      onStatClick(key, op, '0', label, vsPair)
    } else {
      // Fallback: fixed value filter
      const numVal = typeof value === 'number' ? value : parseFloat(value)
      if (isNaN(numVal)) return
      const op = higherBetter ? '>=' : '<='
      onStatClick(key, op, numVal.toString(), label)
    }
  }

  return (
    <table className="matchup-table">
      <thead>
        <tr>
          <th className="mu-rank">Rank</th>
          <th className="mu-val">{away.name || 'Away'}</th>
          <th className="mu-label">Stat</th>
          <th className="mu-val">{home.name || 'Home'}</th>
          <th className="mu-rank">Rank</th>
        </tr>
      </thead>
      <tbody>
        {MATCHUP_STATS.map(({ key, label, higherBetter }) => {
          const aVal = away[key]
          const hVal = home[key]
          const aNum = typeof aVal === 'number' ? aVal : null
          const hNum = typeof hVal === 'number' ? hVal : null

          let aClass = ''
          let hClass = ''
          if (aNum != null && hNum != null && aNum !== hNum) {
            const aBetter = higherBetter ? aNum > hNum : aNum < hNum
            aClass = aBetter ? 'stat-better' : 'stat-worse'
            hClass = aBetter ? 'stat-worse' : 'stat-better'
          }

          const fmt = (v: any) => {
            if (v == null) return '-'
            if (key.includes('pct') && typeof v === 'number' && v <= 1) return `${(v * 100).toFixed(1)}%`
            if (key.includes('pct') && typeof v === 'number') return `${v.toFixed(1)}%`
            if (typeof v === 'number') return v.toFixed(1)
            return String(v)
          }

          const clickable = onStatClick ? ' stat-clickable' : ''

          const aRank = awayRanks[key]
          const hRank = homeRanks[key]

          return (
            <tr key={key}>
              <td className="mu-rank">{aRank != null ? `#${aRank}` : ''}</td>
              <td
                className={'mu-val ' + aClass + clickable}
                onClick={() => handleStatClick(key, aVal, higherBetter, label)}
                title={onStatClick && aVal != null ? `Click to add ${label} filter` : undefined}
              >
                {fmt(aVal)}
              </td>
              <td className="mu-label">{label}</td>
              <td
                className={'mu-val ' + hClass + clickable}
                onClick={() => handleStatClick(key, hVal, higherBetter, label)}
                title={onStatClick && hVal != null ? `Click to add ${label} filter` : undefined}
              >
                {fmt(hVal)}
              </td>
              <td className="mu-rank">{hRank != null ? `#${hRank}` : ''}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// ============================================================================
// GameBacktest Component — inline backtest from a game card
// ============================================================================

function GameBacktest({ game }: { game: Game }) {
  const [team, setTeam] = useState<string>(game.home.name)
  const [filters, setFilters] = useState<Filter[]>([])
  const [columns, setColumns] = useState<ColumnGroups>({})
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [games, setGames] = useState<BacktestGame[]>([])
  const [pnl, setPnl] = useState<PnlPoint[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    axios.get(`${API_BASE}/api/columns`).then(r => setColumns(r.data)).catch(() => {})
  }, [])

  const addStatFilter = (stat: string, op: string, value: string, _label: string, compare_col?: string) => {
    // Don't add duplicate stat filters
    const newFilter: Filter = { stat, op, value, compare_col }
    if (filters.some(f => f.stat === stat)) {
      setFilters(filters.map(f => f.stat === stat ? newFilter : f))
    } else {
      setFilters([...filters, newFilter])
    }
  }

  const removeFilter = (i: number) => setFilters(filters.filter((_, idx) => idx !== i))

  const updateFilter = (i: number, field: keyof Filter, value: string) => {
    const next = [...filters]
    next[i] = { ...next[i], [field]: value }
    setFilters(next)
  }

  const addFilter = () => setFilters([...filters, { stat: '', op: '>', value: '' }])

  const runBacktest = () => {
    setLoading(true)
    const validFilters = filters.filter(f => f.stat && f.value)
    axios.post(`${API_BASE}/api/backtest`, {
      team: team || null,
      filters: validFilters,
      name: team ? `${team} — custom` : 'Both teams — custom',
    })
      .then(r => {
        setResult(r.data.result)
        setGames(r.data.games || [])
        setPnl(r.data.pnl || [])
      })
      .finally(() => setLoading(false))
  }

  return (
    <div className="game-backtest">
      <div className="section-header">Matchup Stats (click a value to add filter)</div>
      <MatchupTable gameId={game.game_id} onStatClick={addStatFilter} />

      <div className="section-header" style={{ marginTop: 16 }}>Quick Backtest</div>

      <div className="game-bt-controls">
        <div className="team-toggle">
          <button
            className={`btn btn-sm ${team === game.away.name ? 'primary' : ''}`}
            onClick={() => setTeam(game.away.name)}
          >
            {game.away.name}
          </button>
          <button
            className={`btn btn-sm ${team === game.home.name ? 'primary' : ''}`}
            onClick={() => setTeam(game.home.name)}
          >
            {game.home.name}
          </button>
          <button
            className={`btn btn-sm ${team === '' ? 'primary' : ''}`}
            onClick={() => setTeam('')}
          >
            All Teams
          </button>
        </div>

        {filters.length > 0 && (
          <div className="game-bt-filters">
            {filters.map((f, i) => (
              <div key={i} className="filter-row">
                <select value={f.stat} onChange={e => {
                  const next = [...filters]
                  next[i] = { ...next[i], stat: e.target.value }
                  // Auto-set compare_col if pairing exists
                  const pair = STAT_VS_PAIRS[e.target.value]
                  if (pair && next[i].compare_col) {
                    next[i].compare_col = pair
                  }
                  setFilters(next)
                }}>
                  <option value="">-- stat --</option>
                  {Object.entries(columns).map(([group, cols]) => (
                    <optgroup key={group} label={group}>
                      {cols.map(c => <option key={c.col} value={c.col}>{c.label}</option>)}
                    </optgroup>
                  ))}
                </select>
                <select className="op" value={f.op} onChange={e => updateFilter(i, 'op', e.target.value)}>
                  <option value=">">{'>'}</option>
                  <option value=">=">{'>='}</option>
                  <option value="<">{'<'}</option>
                  <option value="<=">{'<='}</option>
                  <option value="==">{'=='}</option>
                  <option value="!=">{'!='}</option>
                </select>
                {f.compare_col ? (
                  <>
                    <VsCompareSelect value={f.compare_col!} onChange={v => {
                      const next = [...filters]
                      next[i] = { ...next[i], compare_col: v }
                      setFilters(next)
                    }} />
                    <span className="vs-label">+</span>
                    <input
                      value={f.value}
                      onChange={e => updateFilter(i, 'value', e.target.value)}
                      placeholder="diff"
                      title="Minimum differential (0 = just beat it)"
                      onKeyDown={e => e.key === 'Enter' && runBacktest()}
                      style={{ width: 40 }}
                    />
                    <button className="btn btn-sm" title="Switch to fixed value" onClick={() => {
                      const next = [...filters]
                      next[i] = { ...next[i], compare_col: undefined, value: '' }
                      setFilters(next)
                    }}>val</button>
                  </>
                ) : (
                  <>
                    <input
                      value={f.value}
                      onChange={e => updateFilter(i, 'value', e.target.value)}
                      placeholder="value"
                      onKeyDown={e => e.key === 'Enter' && runBacktest()}
                    />
                    {STAT_VS_PAIRS[f.stat] && (
                      <button className="btn btn-sm" title={`Compare vs ${STAT_VS_PAIRS[f.stat]}`} onClick={() => {
                        const next = [...filters]
                        next[i] = { ...next[i], compare_col: STAT_VS_PAIRS[f.stat], value: '0' }
                        setFilters(next)
                      }}>vs</button>
                    )}
                  </>
                )}
                <button className="btn btn-sm btn-danger" onClick={() => removeFilter(i)}>x</button>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 4, marginTop: 6 }}>
          <button className="btn btn-sm" onClick={addFilter}>+ Filter</button>
          <button className="btn btn-sm primary" onClick={runBacktest} disabled={loading}>
            {loading ? 'Running...' : 'Run Backtest'}
          </button>
          {filters.length > 0 && (
            <button className="btn btn-sm btn-danger" onClick={() => setFilters([])}>Clear All</button>
          )}
        </div>
      </div>

      {result && (
        <div className="game-bt-results">
          {result.n === 0 ? (
            <div className="no-games" style={{ padding: 16 }}>No games matched these filters.</div>
          ) : (
            <>
              <div className="result-summary">
                <div className="result-stat">
                  <div className="label">Record</div>
                  <div className="value">{result.wins}-{result.losses}</div>
                </div>
                <div className="result-stat">
                  <div className="label">Games</div>
                  <div className="value">{result.n}</div>
                </div>
                <div className="result-stat">
                  <div className="label">ATS %</div>
                  <div className={`value ${result.profitable ? 'positive' : 'negative'}`}>
                    {result.win_pct != null ? `${(result.win_pct * 100).toFixed(1)}%` : '-'}
                  </div>
                </div>
                <div className="result-stat">
                  <div className="label">ROI</div>
                  <div className={`value ${(result.roi ?? 0) > 0 ? 'positive' : 'negative'}`}>
                    {result.roi != null ? `${result.roi > 0 ? '+' : ''}${result.roi.toFixed(1)}%` : '-'}
                  </div>
                </div>
                <div className="result-stat">
                  <div className="label">p-value</div>
                  <div className={`value ${(result.p_value ?? 1) < 0.05 ? 'positive' : ''}`}>
                    {result.p_value?.toFixed(4) || '-'}
                  </div>
                </div>
                <div className="result-stat">
                  <div className="label">95% CI</div>
                  <div className="value">
                    {result.ci_low != null ? `${(result.ci_low * 100).toFixed(1)}%-${(result.ci_high! * 100).toFixed(1)}%` : '-'}
                  </div>
                </div>
              </div>

              {pnl.length > 0 && <PnLChart data={pnl} />}

              {games.length > 0 && (
                <details className="games-details">
                  <summary className="section-header games-summary">
                    Games ({games.length}) — click to expand
                  </summary>
                  <table className="games-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Team</th>
                        <th>Opp</th>
                        <th>Spread</th>
                        <th>Score</th>
                        <th>ATS</th>
                        <th>Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {games.map((g, i) => (
                        <tr key={i}>
                          <td>{g.date}</td>
                          <td>{g.team}</td>
                          <td>{g.opponent}</td>
                          <td>{formatSpread(g.spread)}</td>
                          <td>{g.team_score}-{g.opp_score}</td>
                          <td>{g.ats_margin > 0 ? '+' : ''}{g.ats_margin.toFixed(1)}</td>
                          <td className={g.covered ? 'covered' : 'missed'}>
                            {g.covered ? 'COVER' : 'MISS'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// OddsMovementChart Component
// ============================================================================

type OddsChartView = 'spread' | 'spread_price' | 'moneyline'

function OddsMovementChart({ gameId }: { gameId: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [oddsData, setOddsData] = useState<any>(null)
  const [chartView, setChartView] = useState<OddsChartView>('spread')

  useEffect(() => {
    axios.get(`${API_BASE}/api/game/${gameId}/odds`)
      .then(r => setOddsData(r.data))
      .catch(() => {})
  }, [gameId])

  useEffect(() => {
    if (!containerRef.current || !oddsData || !oddsData.timeline?.length) return

    const spreadData: Array<{ time: number; value: number }> = []
    const homeSpreadPrice: Array<{ time: number; value: number }> = []
    const awaySpreadPrice: Array<{ time: number; value: number }> = []
    const mlHomeData: Array<{ time: number; value: number }> = []
    const mlAwayData: Array<{ time: number; value: number }> = []

    for (const snap of oddsData.timeline) {
      const t = Math.floor(new Date(snap.time).getTime() / 1000)
      if (snap.home_spread != null) spreadData.push({ time: t, value: snap.home_spread })
      if (snap.home_spread_price != null) homeSpreadPrice.push({ time: t, value: snap.home_spread_price })
      if (snap.away_spread_price != null) awaySpreadPrice.push({ time: t, value: snap.away_spread_price })
      if (snap.home_ml != null) mlHomeData.push({ time: t, value: snap.home_ml })
      if (snap.away_ml != null) mlAwayData.push({ time: t, value: snap.away_ml })
    }

    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 220,
      layout: {
        background: { color: '#fdf6e3' },
        textColor: '#586e75',
        fontFamily: "'Fira Code', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: '#eee8d5' },
        horzLines: { color: '#eee8d5' },
      },
      timeScale: { timeVisible: true },
      rightPriceScale: { visible: true },
      leftPriceScale: { visible: false },
    })
    chartRef.current = chart

    if (chartView === 'spread' && spreadData.length > 0) {
      const s = chart.addSeries(LineSeries, {
        color: 'rgb(50, 50, 255)',
        lineWidth: 2,
        title: `${oddsData.home} Spread`,
      })
      s.setData(spreadData as any)
    }

    if (chartView === 'spread_price') {
      if (homeSpreadPrice.length > 0) {
        const s = chart.addSeries(LineSeries, {
          color: 'rgb(50, 50, 255)',
          lineWidth: 2,
          title: `${oddsData.home} Price`,
        })
        s.setData(homeSpreadPrice as any)
      }
      if (awaySpreadPrice.length > 0) {
        const s = chart.addSeries(LineSeries, {
          color: 'rgb(255, 50, 150)',
          lineWidth: 2,
          title: `${oddsData.away} Price`,
        })
        s.setData(awaySpreadPrice as any)
      }
    }

    if (chartView === 'moneyline') {
      if (mlHomeData.length > 0) {
        const s = chart.addSeries(LineSeries, {
          color: 'rgb(50, 50, 255)',
          lineWidth: 2,
          title: `${oddsData.home} ML`,
        })
        s.setData(mlHomeData as any)
      }
      if (mlAwayData.length > 0) {
        const s = chart.addSeries(LineSeries, {
          color: 'rgb(255, 50, 150)',
          lineWidth: 2,
          title: `${oddsData.away} ML`,
        })
        s.setData(mlAwayData as any)
      }
    }

    chart.timeScale().fitContent()

    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [oddsData, chartView])

  if (!oddsData || !oddsData.timeline?.length) {
    return <div style={{ fontSize: 11, color: '#657b83', marginTop: 8 }}>No odds movement data</div>
  }

  const tl = oddsData.timeline
  return (
    <div>
      <div className="section-header">Odds Movement — {oddsData.home} vs {oddsData.away} ({oddsData.snapshot_count} snapshots)</div>
      <div className="chart-view-toggle">
        <button className={chartView === 'spread' ? 'active' : ''} onClick={() => setChartView('spread')}>Spread</button>
        <button className={chartView === 'spread_price' ? 'active' : ''} onClick={() => setChartView('spread_price')}>Spread Price</button>
        <button className={chartView === 'moneyline' ? 'active' : ''} onClick={() => setChartView('moneyline')}>Moneyline</button>
      </div>
      <div ref={containerRef} className="odds-chart-container" />
      <div className="odds-timeline-table" style={{ maxHeight: 250, overflowY: 'auto' }}>
        <table className="matchup-table" style={{ fontSize: 11 }}>
          <thead>
            <tr>
              <th>Time</th>
              <th>Spread</th>
              <th>{oddsData.home} ML</th>
              <th>{oddsData.away} ML</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {tl.map((s: any, i: number) => (
              <tr key={i}>
                <td style={{ whiteSpace: 'nowrap' }}>{new Date(s.time).toLocaleString('en-US', { month: 'numeric', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true })}</td>
                <td>{s.home_spread != null ? `${formatSpread(s.home_spread)} (${formatML(s.home_spread_price)})` : '-'}</td>
                <td className={s.home_ml != null && s.home_ml < 0 ? 'fav' : 'dog'}>{s.home_ml != null ? formatML(s.home_ml) : '-'}</td>
                <td className={s.away_ml != null && s.away_ml < 0 ? 'fav' : 'dog'}>{s.away_ml != null ? formatML(s.away_ml) : '-'}</td>
                <td>{s.total != null ? `${s.total} (O ${formatML(s.over_price)} / U ${formatML(s.under_price)})` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ============================================================================
// GameCard Component
// ============================================================================

function GameCard({ game }: { game: Game }) {
  const [expanded, setExpanded] = useState(false)

  const odds = game.odds
  const hasScores = game.home.score != null

  return (
    <div className={`game-card ${expanded ? 'expanded' : ''}`} onClick={() => !expanded && setExpanded(true)} style={{ cursor: expanded ? 'default' : 'pointer' }}>
      <table className="odds-table">
        <thead>
          <tr>
            <th className="ot-meta"></th>
            <th className="ot-team"></th>
            {hasScores && <th className="ot-score">Score</th>}
            <th className="ot-line">Spread</th>
            <th className="ot-line">Win</th>
            {odds?.total != null && <th className="ot-line">Total</th>}
          </tr>
        </thead>
        <tbody>
          {/* Away row */}
          <tr className="ot-row">
            <td className="ot-meta" rowSpan={2}>
              <div className="ot-date">{formatTime(game.commence_time, (game as any).date)}</div>
              <div className={`game-status ${getStatusClass(game.status)}`}>
                {getStatusLabel(game.status, game.status_detail || '')}
              </div>
              {game.neutral_site && <div className="ot-neutral">Neutral</div>}
            </td>
            <td className="ot-team">
              <div className="team-name">{game.away.display_name}</div>
              <div className="team-record">
                {game.away.stats?.record || ''} {game.away.stats?.ats ? `(${game.away.stats.ats} ATS)` : ''}
              </div>
            </td>
            {hasScores && <td className="ot-score">{game.away.score}</td>}
            <td className="ot-line">
              {odds && (
                <span className={odds.away_spread != null && odds.away_spread < 0 ? 'fav' : 'dog'}>
                  {formatSpread(odds.away_spread)}
                  {odds.away_spread_price != null && <span className="odds-juice"> ({formatML(odds.away_spread_price)})</span>}
                </span>
              )}
            </td>
            <td className="ot-line">
              {odds && (
                <span className={odds.away_ml != null && odds.away_ml < 0 ? 'fav' : 'dog'}>
                  {formatML(odds.away_ml)}
                </span>
              )}
            </td>
            {odds?.total != null && (
              <td className="ot-line">
                <span>O {odds.total}{odds.over_price != null && <span className="odds-juice"> ({formatML(odds.over_price)})</span>}</span>
              </td>
            )}
          </tr>
          {/* Home row */}
          <tr className="ot-row">
            <td className="ot-team">
              <div className="team-name">{game.home.display_name}</div>
              <div className="team-record">
                {game.home.stats?.record || ''} {game.home.stats?.ats ? `(${game.home.stats.ats} ATS)` : ''}
              </div>
            </td>
            {hasScores && <td className="ot-score">{game.home.score}</td>}
            <td className="ot-line">
              {odds && (
                <span className={odds.spread != null && odds.spread < 0 ? 'fav' : 'dog'}>
                  {formatSpread(odds.spread)}
                  {odds.spread_price != null && <span className="odds-juice"> ({formatML(odds.spread_price)})</span>}
                </span>
              )}
            </td>
            <td className="ot-line">
              {odds && (
                <span className={odds.ml != null && odds.ml < 0 ? 'fav' : 'dog'}>
                  {formatML(odds.ml)}
                </span>
              )}
            </td>
            {odds?.total != null && (
              <td className="ot-line">
                <span>U {odds.total}{odds.under_price != null && <span className="odds-juice"> ({formatML(odds.under_price)})</span>}</span>
              </td>
            )}
          </tr>
        </tbody>
      </table>
      {odds?.bookmaker && (
        <div className="odds-source">via {odds.bookmaker}</div>
      )}

      {expanded && (
        <div className="game-detail" onClick={e => e.stopPropagation()}>
          <button className="collapse-btn" onClick={e => { e.stopPropagation(); setExpanded(false) }}>collapse</button>
          <GameBacktest game={game} />
          <OddsMovementChart gameId={game.game_id} />
        </div>
      )}
    </div>
  )
}

// ============================================================================
// TodayDashboard Component
// ============================================================================

function TodayDashboard() {
  const todayStr = new Date().toISOString().slice(0, 10)
  const [selectedDate, setSelectedDate] = useState(todayStr)
  const [data, setData] = useState<TodayData | null>(null)
  const [gameDates, setGameDates] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  // Fetch list of all dates with games
  useEffect(() => {
    axios.get(`${API_BASE}/api/dates`).then(r => setGameDates(r.data)).catch(() => {})
  }, [])

  // Fetch games for selected date
  const fetchGames = useCallback(() => {
    setLoading(true)
    if (selectedDate === todayStr) {
      // Use live today endpoint for current date
      axios.get(`${API_BASE}/api/today`)
        .then(r => { setData(r.data); setLoading(false) })
        .catch(() => {
          // Fallback to game logs if today.json is empty
          axios.get(`${API_BASE}/api/games/${selectedDate}`)
            .then(r => { setData(r.data); setLoading(false) })
            .catch(() => setLoading(false))
        })
    } else {
      axios.get(`${API_BASE}/api/games/${selectedDate}`)
        .then(r => { setData(r.data); setLoading(false) })
        .catch(() => setLoading(false))
    }
  }, [selectedDate, todayStr])

  useEffect(() => {
    fetchGames()
    // Only auto-refresh if viewing today
    if (selectedDate === todayStr) {
      const interval = setInterval(fetchGames, 60000)
      return () => clearInterval(interval)
    }
  }, [fetchGames, selectedDate, todayStr])

  const shiftDate = (days: number) => {
    const d = new Date(selectedDate + 'T12:00:00')
    d.setDate(d.getDate() + days)
    setSelectedDate(d.toISOString().slice(0, 10))
  }

  const jumpToNearestGame = (direction: number) => {
    if (gameDates.length === 0) return
    const idx = gameDates.indexOf(selectedDate)
    if (direction < 0) {
      // Find the previous date with games
      const earlier = gameDates.filter(d => d < selectedDate)
      if (earlier.length > 0) setSelectedDate(earlier[earlier.length - 1])
    } else {
      const later = gameDates.filter(d => d > selectedDate)
      if (later.length > 0) setSelectedDate(later[0])
    }
  }

  return (
    <div>
      <div className="date-nav" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <button onClick={() => jumpToNearestGame(-1)} title="Previous date with games">&laquo;</button>
        <button onClick={() => shiftDate(-1)}>&lsaquo;</button>
        <input
          type="date"
          value={selectedDate}
          onChange={e => setSelectedDate(e.target.value)}
          style={{ fontFamily: 'inherit', fontSize: 'inherit', padding: '0.25rem 0.5rem' }}
        />
        <button onClick={() => shiftDate(1)}>&rsaquo;</button>
        <button onClick={() => jumpToNearestGame(1)} title="Next date with games">&raquo;</button>
        <button onClick={() => setSelectedDate(todayStr)}>Today</button>
        {data && <span style={{ opacity: 0.6 }}>{data.game_count} game{data.game_count !== 1 ? 's' : ''}</span>}
      </div>

      {loading ? (
        <div className="loading">Loading games...</div>
      ) : !data || data.games.length === 0 ? (
        <div className="no-games">No games found for {selectedDate}.</div>
      ) : (
        <div className="games-grid">
          {data.games.map(game => (
            <GameCard key={game.game_id} game={game} />
          ))}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// PnLChart Component
// ============================================================================

function PnLChart({ data }: { data: PnlPoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !data.length) return

    if (chartRef.current) {
      chartRef.current.remove()
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 250,
      layout: {
        background: { color: '#fdf6e3' },
        textColor: '#586e75',
        fontFamily: "'Fira Code', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: '#eee8d5' },
        horzLines: { color: '#eee8d5' },
      },
    })
    chartRef.current = chart

    const series = chart.addSeries(LineSeries, {
      color: '#268bd2',
      lineWidth: 2,
    })

    // Deduplicate dates (keep last pnl per date)
    const dateMap = new Map<string, number>()
    for (const p of data) {
      dateMap.set(p.date, p.pnl)
    }
    const chartData = Array.from(dateMap.entries()).map(([date, pnl]) => ({
      time: date as any,
      value: pnl,
    }))

    series.setData(chartData)
    chart.timeScale().fitContent()

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [data])

  return <div ref={containerRef} className="pnl-chart" />
}

// ============================================================================
// BacktestPage Component
// ============================================================================

function BacktestPage() {
  const [columns, setColumns] = useState<ColumnGroups>({})
  const [filters, setFilters] = useState<Filter[]>([{ stat: '', op: '>', value: '' }])
  const [team, setTeam] = useState<string>('')
  const [teams, setTeams] = useState<string[]>([])
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [games, setGames] = useState<BacktestGame[]>([])
  const [pnl, setPnl] = useState<PnlPoint[]>([])
  const [strategies, setStrategies] = useState<BacktestResult[]>([])
  const [scanResults, setScanResults] = useState<BacktestResult[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'custom' | 'strategies' | 'scan'>('custom')
  const [sortCol, setSortCol] = useState<string>('win_pct')
  const [sortAsc, setSortAsc] = useState(false)

  useEffect(() => {
    axios.get(`${API_BASE}/api/columns`).then(r => setColumns(r.data)).catch(() => {})
    axios.get(`${API_BASE}/api/teams`).then(r => {
      setTeams(r.data.map((t: any) => t.name).sort())
    }).catch(() => {})
  }, [])

  const addFilter = () => setFilters([...filters, { stat: '', op: '>', value: '' }])
  const removeFilter = (i: number) => setFilters(filters.filter((_, idx) => idx !== i))
  const updateFilter = (i: number, field: keyof Filter, value: string) => {
    const next = [...filters]
    next[i] = { ...next[i], [field]: value }
    setFilters(next)
  }

  const runBacktest = () => {
    setLoading(true)
    const validFilters = filters.filter(f => f.stat && f.value)
    axios.post(`${API_BASE}/api/backtest`, {
      team: team || null,
      filters: validFilters,
      name: team ? `Custom: ${team}` : 'Custom backtest',
    })
      .then(r => {
        setResult(r.data.result)
        setGames(r.data.games || [])
        setPnl(r.data.pnl || [])
        setActiveTab('custom')
      })
      .finally(() => setLoading(false))
  }

  const runStrategies = () => {
    setLoading(true)
    axios.get(`${API_BASE}/api/backtest/strategies`)
      .then(r => { setStrategies(r.data); setActiveTab('strategies') })
      .finally(() => setLoading(false))
  }

  const runScan = () => {
    setLoading(true)
    axios.post(`${API_BASE}/api/backtest/scan`, { min_games: 20 })
      .then(r => { setScanResults(r.data); setActiveTab('scan') })
      .finally(() => setLoading(false))
  }

  const sortTable = (col: string) => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
  }

  const sortedData = (data: BacktestResult[]) => {
    return [...data].sort((a, b) => {
      const aVal = (a as any)[sortCol] ?? 0
      const bVal = (b as any)[sortCol] ?? 0
      return sortAsc ? aVal - bVal : bVal - aVal
    })
  }

  const renderStrategyTable = (data: BacktestResult[]) => (
    <table className="strategy-table">
      <thead>
        <tr>
          <th onClick={() => sortTable('name')}>Strategy</th>
          <th onClick={() => sortTable('n')}>N</th>
          <th onClick={() => sortTable('wins')}>W-L</th>
          <th onClick={() => sortTable('win_pct')}>ATS%</th>
          <th onClick={() => sortTable('roi')}>ROI</th>
          <th onClick={() => sortTable('p_value')}>p-val</th>
        </tr>
      </thead>
      <tbody>
        {sortedData(data).map((r, i) => (
          <tr key={i}>
            <td className={r.profitable ? 'profitable' : ''}>
              {r.profitable ? '+ ' : '  '}{r.name}
            </td>
            <td>{r.n}</td>
            <td>{r.wins}-{r.losses}</td>
            <td className={r.profitable ? 'profitable' : ''}>
              {r.win_pct != null ? `${(r.win_pct * 100).toFixed(1)}%` : '-'}
            </td>
            <td className={(r.roi ?? 0) > 0 ? 'profitable' : 'missed'}>
              {r.roi != null ? `${r.roi > 0 ? '+' : ''}${r.roi.toFixed(1)}%` : '-'}
            </td>
            <td className={(r.p_value ?? 1) < 0.05 ? 'significant' : ''}>
              {r.p_value != null ? r.p_value.toFixed(4) : '-'}
              {(r.p_value ?? 1) < 0.01 ? ' ***' : (r.p_value ?? 1) < 0.05 ? ' **' : (r.p_value ?? 1) < 0.1 ? ' *' : ''}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )

  return (
    <div className="backtest-container">
      <div className="backtest-sidebar">
        <div className="section-header">Team</div>
        <select className="team-select" value={team} onChange={e => setTeam(e.target.value)}>
          <option value="">All Teams</option>
          {teams.map(t => <option key={t} value={t}>{t}</option>)}
        </select>

        <div className="section-header">Filters</div>
        {filters.map((f, i) => (
          <div key={i} className="filter-row">
            <select value={f.stat} onChange={e => {
              const next = [...filters]
              next[i] = { ...next[i], stat: e.target.value }
              const pair = STAT_VS_PAIRS[e.target.value]
              if (pair && next[i].compare_col) next[i].compare_col = pair
              setFilters(next)
            }}>
              <option value="">-- stat --</option>
              {Object.entries(columns).map(([group, cols]) => (
                <optgroup key={group} label={group}>
                  {cols.map(c => <option key={c.col} value={c.col}>{c.label}</option>)}
                </optgroup>
              ))}
            </select>
            <select className="op" value={f.op} onChange={e => updateFilter(i, 'op', e.target.value)}>
              <option value=">">{'>'}</option>
              <option value=">=">{'>='}</option>
              <option value="<">{'<'}</option>
              <option value="<=">{'<='}</option>
              <option value="==">{'=='}</option>
              <option value="!=">{'!='}</option>
            </select>
            {f.compare_col ? (
              <>
                <span className="vs-label">{f.compare_col}</span>
                <span className="vs-label">+</span>
                <input
                  value={f.value}
                  onChange={e => updateFilter(i, 'value', e.target.value)}
                  placeholder="diff"
                  title="Minimum differential"
                  onKeyDown={e => e.key === 'Enter' && runBacktest()}
                  style={{ width: 40 }}
                />
                <button className="btn btn-sm" title="Switch to fixed value" onClick={() => {
                  const next = [...filters]
                  next[i] = { ...next[i], compare_col: undefined, value: '' }
                  setFilters(next)
                }}>val</button>
              </>
            ) : (
              <>
                <input
                  value={f.value}
                  onChange={e => updateFilter(i, 'value', e.target.value)}
                  placeholder="value"
                  onKeyDown={e => e.key === 'Enter' && runBacktest()}
                />
                {STAT_VS_PAIRS[f.stat] && (
                  <button className="btn btn-sm" title={`Compare vs ${STAT_VS_PAIRS[f.stat]}`} onClick={() => {
                    const next = [...filters]
                    next[i] = { ...next[i], compare_col: STAT_VS_PAIRS[f.stat], value: '0' }
                    setFilters(next)
                  }}>vs</button>
                )}
              </>
            )}
            <button className="btn btn-sm btn-danger" onClick={() => removeFilter(i)}>x</button>
          </div>
        ))}
        <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
          <button className="btn btn-sm" onClick={addFilter}>+ Filter</button>
        </div>

        <div style={{ display: 'flex', gap: 4, marginTop: 12, flexWrap: 'wrap' }}>
          <button className="btn primary" onClick={runBacktest} disabled={loading}>
            {loading ? 'Running...' : 'Run Backtest'}
          </button>
          <button className="btn" onClick={runStrategies} disabled={loading}>
            Built-in Strategies
          </button>
          <button className="btn" onClick={runScan} disabled={loading}>
            Scan Edges
          </button>
        </div>
      </div>

      <div className="backtest-main">
        {activeTab === 'custom' && result && (
          <>
            <div className="section-header">{result.name}</div>
            {result.n === 0 ? (
              <div className="no-games">No games matched these filters.</div>
            ) : (
              <>
                <div className="result-summary">
                  <div className="result-stat">
                    <div className="label">Record</div>
                    <div className="value">{result.wins}-{result.losses}</div>
                  </div>
                  <div className="result-stat">
                    <div className="label">Games</div>
                    <div className="value">{result.n}</div>
                  </div>
                  <div className="result-stat">
                    <div className="label">ATS %</div>
                    <div className={`value ${result.profitable ? 'positive' : 'negative'}`}>
                      {result.win_pct != null ? `${(result.win_pct * 100).toFixed(1)}%` : '-'}
                    </div>
                  </div>
                  <div className="result-stat">
                    <div className="label">ROI</div>
                    <div className={`value ${(result.roi ?? 0) > 0 ? 'positive' : 'negative'}`}>
                      {result.roi != null ? `${result.roi > 0 ? '+' : ''}${result.roi.toFixed(1)}%` : '-'}
                    </div>
                  </div>
                  <div className="result-stat">
                    <div className="label">p-value</div>
                    <div className={`value ${(result.p_value ?? 1) < 0.05 ? 'positive' : ''}`}>
                      {result.p_value?.toFixed(4) || '-'}
                    </div>
                  </div>
                  <div className="result-stat">
                    <div className="label">95% CI</div>
                    <div className="value">
                      {result.ci_low != null ? `${(result.ci_low * 100).toFixed(1)}% - ${(result.ci_high! * 100).toFixed(1)}%` : '-'}
                    </div>
                  </div>
                </div>

                {pnl.length > 0 && (
                  <>
                    <div className="section-header">Cumulative P&L (units)</div>
                    <PnLChart data={pnl} />
                  </>
                )}

                {games.length > 0 && (
                  <>
                    <div className="section-header">Games ({games.length})</div>
                    <table className="games-table">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Team</th>
                          <th>Opp</th>
                          <th>H/A</th>
                          <th>Spread</th>
                          <th>Score</th>
                          <th>Margin</th>
                          <th>ATS</th>
                          <th>Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {games.map((g, i) => (
                          <tr key={i}>
                            <td>{g.date}</td>
                            <td>{g.team}</td>
                            <td>{g.opponent}</td>
                            <td>{g.home_away}</td>
                            <td>{formatSpread(g.spread)}</td>
                            <td>{g.team_score}-{g.opp_score}</td>
                            <td>{g.margin > 0 ? '+' : ''}{g.margin}</td>
                            <td>{g.ats_margin > 0 ? '+' : ''}{g.ats_margin.toFixed(1)}</td>
                            <td className={g.covered ? 'covered' : 'missed'}>
                              {g.covered ? 'COVER' : 'MISS'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}
              </>
            )}
          </>
        )}

        {activeTab === 'strategies' && strategies.length > 0 && (
          <>
            <div className="section-header">Built-in Strategies ({strategies.length})</div>
            {renderStrategyTable(strategies)}
          </>
        )}

        {activeTab === 'scan' && scanResults.length > 0 && (
          <>
            <div className="section-header">Edge Scan ({scanResults.length} differentials)</div>
            {renderStrategyTable(scanResults)}
          </>
        )}

        {!result && strategies.length === 0 && scanResults.length === 0 && !loading && (
          <div className="no-games">
            Configure filters and run a backtest, or try the built-in strategies.
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// App Component
// ============================================================================

function App() {
  const [view, setView] = useState<View>('today')
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)

  useEffect(() => {
    axios.get(`${API_BASE}/api/today`)
      .then(r => setUpdatedAt(r.data.updated_at))
      .catch(() => {})
  }, [])

  return (
    <div className="app-container">
      <div className="app-header">
        <h1>NCAAB</h1>
        <button
          className={`nav-btn ${view === 'today' ? 'active' : ''}`}
          onClick={() => setView('today')}
        >
          Today
        </button>
        <button
          className={`nav-btn ${view === 'backtest' ? 'active' : ''}`}
          onClick={() => setView('backtest')}
        >
          Backtest
        </button>
        <div className="header-spacer" />
        {updatedAt && (
          <span className="updated-at">
            Updated: {new Date(updatedAt).toLocaleTimeString()}
          </span>
        )}
      </div>
      <div className="main-content">
        {view === 'today' && <TodayDashboard />}
        {view === 'backtest' && <BacktestPage />}
      </div>
    </div>
  )
}

export default App
