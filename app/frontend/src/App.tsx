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

type View = 'scores' | 'team' | 'backtest' | 'trends'

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
  { group: 'Ranks', cols: [
    { col: 'rank_opp_win_pct', label: 'Opp Win % Rank' },
    { col: 'rank_opp_ats_win_pct', label: 'Opp ATS % Rank' },
    { col: 'rank_opp_ppg', label: 'Opp PPG Rank' },
    { col: 'rank_opp_ft_pct', label: 'Opp FT % Rank' },
    { col: 'rank_opp_3pt_pct', label: 'Opp 3PT % Rank' },
    { col: 'rank_opp_2pt_pct', label: 'Opp 2PT % Rank' },
    { col: 'rank_opp_def_3pt_pct', label: 'Opp Def 3PT % Rank' },
    { col: 'rank_opp_def_2pt_pct', label: 'Opp Def 2PT % Rank' },
    { col: 'rank_opp_oreb_pg', label: 'Opp OREB/G Rank' },
    { col: 'rank_opp_dreb_pg', label: 'Opp DREB/G Rank' },
    { col: 'rank_opp_to_pg', label: 'Opp TO/G Rank' },
    { col: 'rank_opp_forced_to_pg', label: 'Opp Forced TO/G Rank' },
    { col: 'rank_opp_pace', label: 'Opp Pace Rank' },
    { col: 'rank_opp_sos', label: 'Opp SOS Rank' },
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

  const handleStatClick = (key: string, teamVal: any, _higherBetter: boolean, label: string, side: 'home' | 'away') => {
    if (teamVal == null || !onStatClick) return
    const oppSide = side === 'home' ? away : home
    // Always compare the same stat: team_X vs opp_X
    const oppKey = key.replace('team_', 'opp_')
    const teamNum = typeof teamVal === 'number' ? teamVal : parseFloat(teamVal)
    const oppVal = oppSide[key] // opponent's same stat (their team_X is our opp_X)
    const oppNum = typeof oppVal === 'number' ? oppVal : parseFloat(oppVal)
    if (!isNaN(teamNum) && !isNaN(oppNum)) {
      const op = teamNum >= oppNum ? '>' : '<'
      onStatClick(key, op, '0', label, oppKey)
    } else {
      const op = _higherBetter ? '>=' : '<='
      onStatClick(key, op, teamNum.toString(), label)
    }
  }

  const handleRankClick = (key: string, rank: number, label: string, side: 'home' | 'away') => {
    if (!onStatClick) return
    const rankCol = `rank_${key}`
    const oppRankCol = rankCol.replace('rank_team_', 'rank_opp_')
    const oppRanks = side === 'home' ? awayRanks : homeRanks
    const oppRank = oppRanks[key]
    // Lower rank = better, so set direction based on actual values
    const op = (oppRank != null && rank <= oppRank) ? '<' : '>'
    onStatClick(rankCol, op, '0', `${label} Rank`, oppRankCol)
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

          const rankClickable = onStatClick && aRank != null ? ' stat-clickable' : ''

          return (
            <tr key={key}>
              <td
                className={'mu-rank' + rankClickable}
                onClick={() => aRank != null && handleRankClick(key, aRank, label, 'away')}
                title={onStatClick && aRank != null ? `Click to add ${label} Rank filter` : undefined}
              >
                {aRank != null ? `#${aRank}` : ''}
              </td>
              <td
                className={'mu-val ' + aClass + clickable}
                onClick={() => handleStatClick(key, aVal, higherBetter, label, 'away')}
                title={onStatClick && aVal != null ? `Click to add ${label} filter` : undefined}
              >
                {fmt(aVal)}
              </td>
              <td className="mu-label">{label}</td>
              <td
                className={'mu-val ' + hClass + clickable}
                onClick={() => handleStatClick(key, hVal, higherBetter, label, 'home')}
                title={onStatClick && hVal != null ? `Click to add ${label} filter` : undefined}
              >
                {fmt(hVal)}
              </td>
              <td
                className={'mu-rank' + (onStatClick && hRank != null ? ' stat-clickable' : '')}
                onClick={() => hRank != null && handleRankClick(key, hRank, label, 'home')}
                title={onStatClick && hRank != null ? `Click to add ${label} Rank filter` : undefined}
              >
                {hRank != null ? `#${hRank}` : ''}
              </td>
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
  const [team, setTeam] = useState<string>('')
  const [filters, setFilters] = useState<Filter[]>([])
  const [columns, setColumns] = useState<ColumnGroups>({})
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [games, setGames] = useState<BacktestGame[]>([])
  const [pnl, setPnl] = useState<PnlPoint[]>([])
  const [loading, setLoading] = useState(false)
  const [betType, setBetType] = useState<BetType>('ats')

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
      bet_type: betType,
    })
      .then(r => {
        setResult(r.data.result)
        setGames((r.data.games || []).sort((a: BacktestGame, b: BacktestGame) => b.date.localeCompare(a.date)))
        setPnl(r.data.pnl || [])
      })
      .finally(() => setLoading(false))
  }

  return (
    <div className="game-backtest">
      <div className="detail-panel">
        <div className="section-header">Matchup Stats (click a value to add filter)</div>
        <MatchupTable gameId={game.game_id} onStatClick={addStatFilter} />
      </div>

      <div className="detail-panel">
        <div className="section-header">Quick Backtest</div>

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
            {filters.map((f, i) => {
              const isSideFilter = ['is_favorite', 'is_underdog', 'is_home', 'is_away', 'is_neutral', 'is_conference'].includes(f.stat)
              return (
              <div key={i} className="filter-row">
                <select value={f.stat} onChange={e => {
                  const next = [...filters]
                  const newStat = e.target.value
                  const isSide = ['is_favorite', 'is_underdog', 'is_home', 'is_away', 'is_neutral', 'is_conference'].includes(newStat)
                  if (isSide) {
                    next[i] = { ...next[i], stat: newStat, op: '==', value: '1', compare_col: undefined }
                  } else {
                    next[i] = { ...next[i], stat: newStat }
                    const pair = STAT_VS_PAIRS[newStat]
                    if (pair && next[i].compare_col) next[i].compare_col = pair
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
                {isSideFilter ? (
                  <span className="vs-label" style={{ padding: '0 8px' }}>= Yes</span>
                ) : (
                  <>
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
                        <button className="btn btn-sm" title="Compare vs opponent stat" onClick={() => {
                          const pair = STAT_VS_PAIRS[f.stat] || f.stat.replace('team_', 'opp_')
                          const next = [...filters]
                          next[i] = { ...next[i], compare_col: pair, value: '0' }
                          setFilters(next)
                        }}>vs</button>
                      </>
                    )}
                  </>
                )}
                <button className="btn btn-sm btn-danger" onClick={() => removeFilter(i)}>x</button>
              </div>
              )
            })}
          </div>
        )}

        <div style={{ display: 'flex', gap: 4, marginTop: 6, alignItems: 'center' }}>
          <button className="btn btn-sm" onClick={addFilter}>+ Filter</button>
          <button className={`trends-filter-btn ${betType === 'ats' ? 'active' : ''}`} onClick={() => setBetType('ats')}>ATS</button>
          <button className={`trends-filter-btn ${betType === 'ml' ? 'active' : ''}`} onClick={() => setBetType('ml')}>ML</button>
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
                <div className="detail-panel">
                  <div className="section-header">Games ({games.length})</div>
                  <div className="scrollable-table">
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
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
      </div>
    </div>
  )
}

// ============================================================================
// OddsMovementChart Component
// ============================================================================

type OddsChartView = 'spread' | 'spread_price' | 'moneyline' | 'total' | 'total_price'

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

    const series: Record<string, Array<{ time: number; value: number }>> = {
      spread: [], homeSpreadPrice: [], awaySpreadPrice: [],
      homeML: [], awayML: [], total: [], overPrice: [], underPrice: [],
    }

    for (const snap of oddsData.timeline) {
      const t = Math.floor(new Date(snap.time).getTime() / 1000)
      if (snap.home_spread != null) series.spread.push({ time: t, value: snap.home_spread })
      if (snap.home_spread_price != null) series.homeSpreadPrice.push({ time: t, value: snap.home_spread_price })
      if (snap.away_spread_price != null) series.awaySpreadPrice.push({ time: t, value: snap.away_spread_price })
      if (snap.home_ml != null) series.homeML.push({ time: t, value: snap.home_ml })
      if (snap.away_ml != null) series.awayML.push({ time: t, value: snap.away_ml })
      if (snap.total != null) series.total.push({ time: t, value: snap.total })
      if (snap.over_price != null) series.overPrice.push({ time: t, value: snap.over_price })
      if (snap.under_price != null) series.underPrice.push({ time: t, value: snap.under_price })
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

    const blue = 'rgb(50, 50, 255)'
    const pink = 'rgb(255, 50, 150)'

    const addLine = (data: Array<{ time: number; value: number }>, color: string, title: string) => {
      if (data.length === 0) return
      const s = chart.addSeries(LineSeries, { color, lineWidth: 2, title })
      s.setData(data as any)
    }

    if (chartView === 'spread') addLine(series.spread, blue, `${oddsData.home} Spread`)
    if (chartView === 'spread_price') {
      addLine(series.homeSpreadPrice, blue, `${oddsData.home} Spread Price`)
      addLine(series.awaySpreadPrice, pink, `${oddsData.away} Spread Price`)
    }
    if (chartView === 'moneyline') {
      addLine(series.homeML, blue, `${oddsData.home} ML`)
      addLine(series.awayML, pink, `${oddsData.away} ML`)
    }
    if (chartView === 'total') addLine(series.total, blue, 'Total')
    if (chartView === 'total_price') {
      addLine(series.overPrice, blue, 'Over Price')
      addLine(series.underPrice, pink, 'Under Price')
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
  const home = oddsData.home
  const away = oddsData.away
  return (
    <div>
      <div className="detail-panel">
        <div className="section-header">Odds Movement — {home} vs {away} ({oddsData.snapshot_count} snapshots)</div>
        <div className="chart-view-toggle">
          <button className={chartView === 'spread' ? 'active' : ''} onClick={() => setChartView('spread')}>Spread</button>
          <button className={chartView === 'spread_price' ? 'active' : ''} onClick={() => setChartView('spread_price')}>Spread Price</button>
          <button className={chartView === 'moneyline' ? 'active' : ''} onClick={() => setChartView('moneyline')}>Moneyline</button>
          <button className={chartView === 'total' ? 'active' : ''} onClick={() => setChartView('total')}>Total</button>
          <button className={chartView === 'total_price' ? 'active' : ''} onClick={() => setChartView('total_price')}>Total Price</button>
        </div>
        <div ref={containerRef} className="odds-chart-container" />
      </div>
      <div className="detail-panel">
        <div className="section-header">Odds Timeline</div>
        <div className="scrollable-table">
        <table className="matchup-table" style={{ fontSize: 10 }}>
          <thead>
            <tr>
              <th>Date</th>
              <th>Time</th>
              <th>Spread</th>
              <th>{home} Price</th>
              <th>{away} Price</th>
              <th>{home} ML</th>
              <th>{away} ML</th>
              <th>Total</th>
              <th>Over</th>
              <th>Under</th>
            </tr>
          </thead>
          <tbody>
            {tl.map((s: any, i: number) => {
              const dt = new Date(s.time)
              return (
                <tr key={i}>
                  <td style={{ whiteSpace: 'nowrap' }}>{dt.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' })}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>{dt.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })}</td>
                  <td>{s.home_spread != null ? formatSpread(s.home_spread) : '-'}</td>
                  <td>{s.home_spread_price != null ? formatML(s.home_spread_price) : '-'}</td>
                  <td>{s.away_spread_price != null ? formatML(s.away_spread_price) : '-'}</td>
                  <td className={s.home_ml != null && s.home_ml < 0 ? 'fav' : 'dog'}>{s.home_ml != null ? formatML(s.home_ml) : '-'}</td>
                  <td className={s.away_ml != null && s.away_ml < 0 ? 'fav' : 'dog'}>{s.away_ml != null ? formatML(s.away_ml) : '-'}</td>
                  <td>{s.total != null ? s.total : '-'}</td>
                  <td>{s.over_price != null ? formatML(s.over_price) : '-'}</td>
                  <td>{s.under_price != null ? formatML(s.under_price) : '-'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        </div>
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
          <div className="detail-section">
            <GameBacktest game={game} />
          </div>
          <div className="detail-section">
            <OddsMovementChart gameId={game.game_id} />
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// CalendarPicker Component
// ============================================================================

function CalendarPicker({
  selectedDate,
  onSelect,
  gameDates,
  todayStr,
}: {
  selectedDate: string
  onSelect: (date: string) => void
  gameDates: Array<{ date: string; games: number }>
  todayStr: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Derive viewed month from selectedDate
  const sel = new Date(selectedDate + 'T12:00:00')
  const [viewYear, setViewYear] = useState(sel.getFullYear())
  const [viewMonth, setViewMonth] = useState(sel.getMonth())

  // Build a lookup: "YYYY-MM-DD" -> game count
  const gameMap = useRef<Record<string, number>>({})
  gameMap.current = {}
  for (const d of gameDates) gameMap.current[d.date] = d.games

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  // Sync viewed month when selectedDate changes externally
  useEffect(() => {
    const d = new Date(selectedDate + 'T12:00:00')
    setViewYear(d.getFullYear())
    setViewMonth(d.getMonth())
  }, [selectedDate])

  const shiftMonth = (delta: number) => {
    let m = viewMonth + delta
    let y = viewYear
    if (m < 0) { m = 11; y-- }
    if (m > 11) { m = 0; y++ }
    setViewMonth(m)
    setViewYear(y)
  }

  // Build calendar grid
  const firstDay = new Date(viewYear, viewMonth, 1).getDay() // 0=Sun
  const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate()
  const prevMonthDays = new Date(viewYear, viewMonth, 0).getDate()

  const cells: Array<{ date: string; day: number; inMonth: boolean }> = []
  // Leading days from previous month
  for (let i = firstDay - 1; i >= 0; i--) {
    const d = prevMonthDays - i
    const m = viewMonth === 0 ? 12 : viewMonth
    const y = viewMonth === 0 ? viewYear - 1 : viewYear
    const ds = `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`
    cells.push({ date: ds, day: d, inMonth: false })
  }
  // Current month
  for (let d = 1; d <= daysInMonth; d++) {
    const ds = `${viewYear}-${String(viewMonth + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`
    cells.push({ date: ds, day: d, inMonth: true })
  }
  // Trailing days
  const trailing = 7 - (cells.length % 7)
  if (trailing < 7) {
    for (let d = 1; d <= trailing; d++) {
      const m = viewMonth === 11 ? 1 : viewMonth + 2
      const y = viewMonth === 11 ? viewYear + 1 : viewYear
      const ds = `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`
      cells.push({ date: ds, day: d, inMonth: false })
    }
  }

  const monthLabel = new Date(viewYear, viewMonth, 1).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
  const selectedLabel = new Date(selectedDate + 'T12:00:00').toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
  const selectedGames = gameMap.current[selectedDate] || 0

  return (
    <div className="cal-wrapper" ref={ref}>
      <button className="cal-trigger" onClick={() => setOpen(!open)}>
        {selectedLabel}{selectedGames > 0 ? ` — ${selectedGames} game${selectedGames !== 1 ? 's' : ''}` : ''}
        {selectedDate === todayStr ? ' (today)' : ''}
        <span className="cal-caret">{open ? '\u25B2' : '\u25BC'}</span>
      </button>
      {open && (
        <div className="cal-dropdown">
          <div className="cal-header">
            <button onClick={() => shiftMonth(-1)}>&lsaquo;</button>
            <span className="cal-month-label">{monthLabel}</span>
            <button onClick={() => shiftMonth(1)}>&rsaquo;</button>
          </div>
          <div className="cal-grid">
            {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(d => (
              <div key={d} className="cal-dow">{d}</div>
            ))}
            {cells.map((c, i) => {
              const games = gameMap.current[c.date] || 0
              const isSelected = c.date === selectedDate
              const isToday = c.date === todayStr
              return (
                <button
                  key={i}
                  className={[
                    'cal-cell',
                    !c.inMonth && 'cal-outside',
                    isSelected && 'cal-selected',
                    isToday && 'cal-today',
                    games > 0 && 'cal-has-games',
                  ].filter(Boolean).join(' ')}
                  onClick={() => { onSelect(c.date); setOpen(false) }}
                >
                  <span className="cal-day">{c.day}</span>
                  {games > 0 && <span className="cal-games">{games}</span>}
                </button>
              )
            })}
          </div>
          <div className="cal-footer">
            <button onClick={() => { onSelect(todayStr); setOpen(false) }}>Today</button>
          </div>
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
  const [gameDates, setGameDates] = useState<Array<{date: string; games: number}>>([])
  const [loading, setLoading] = useState(true)

  // Fetch list of all dates with games (historical + upcoming)
  useEffect(() => {
    Promise.all([
      axios.get(`${API_BASE}/api/dates`).then(r => r.data as Array<{date: string; games: number}>).catch(() => []),
      axios.get(`${API_BASE}/api/today`).then(r => {
        const games = r.data?.games || []
        const counts: Record<string, number> = {}
        for (const g of games) {
          const gd = g.game_date || (g.commence_time || g.date || '').slice(0, 10)
          if (gd) counts[gd] = (counts[gd] || 0) + 1
        }
        return Object.entries(counts).map(([date, games]) => ({ date, games }))
      }).catch(() => []),
    ]).then(([historical, upcoming]) => {
      const map: Record<string, number> = {}
      for (const h of historical) map[h.date] = h.games
      for (const u of upcoming) if (!map[u.date]) map[u.date] = u.games
      const all = Object.entries(map).map(([date, games]) => ({ date, games })).sort((a, b) => a.date.localeCompare(b.date))
      setGameDates(all)
      // If today has no games, jump to next date with games
      const allDates = all.map(d => d.date)
      if (!allDates.includes(todayStr) && all.length > 0) {
        const next = all.find(d => d.date >= todayStr)
        if (next) setSelectedDate(next.date)
      }
    })
  }, [todayStr])

  // Normalize odds from the /api/today format (nested bookmakers) to the
  // flat format the GameCard expects (spread_price, away_spread_price, etc.)
  const normalizeOdds = (game: any) => {
    const odds = game.odds
    if (!odds || odds.bookmaker) return game // already normalized

    // Find the home/away bookmaker entries by looking for matching spread data
    const bookPref = ['Bovada', 'DraftKings', 'FanDuel', 'BetMGM', 'BetRivers', 'Caesars']
    const bookmakers = odds.bookmakers || []
    let chosen: any = bookmakers[0]
    for (const pref of bookPref) {
      const found = bookmakers.find((b: any) => b.name === pref)
      if (found) { chosen = found; break }
    }
    if (!chosen) return game

    const mkts = chosen.markets || {}
    const homeSpread = mkts.spread?.point ?? odds.spread
    const homeSpreadPrice = mkts.spread?.price ?? null
    const awaySpread = homeSpread != null ? -homeSpread : odds.away_spread
    // Find away spread price from the same bookmaker — look for matching away entry
    // The schedule refresher doesn't store away spread price per bookmaker,
    // so we mirror the home price as an approximation
    const awaySpreadPrice = homeSpreadPrice != null ? homeSpreadPrice : null

    return {
      ...game,
      odds: {
        spread: homeSpread,
        spread_price: homeSpreadPrice,
        away_spread: awaySpread,
        away_spread_price: awaySpreadPrice,
        ml: mkts.ml ?? odds.ml,
        away_ml: odds.away_ml,
        total: mkts.total ?? odds.total,
        over_price: null,
        under_price: null,
        bookmaker: chosen.name,
      },
    }
  }

  // Fetch games for selected date
  const isInitialLoad = useRef(true)
  const fetchGames = useCallback((silent = false) => {
    if (!silent) setLoading(true)
    axios.get(`${API_BASE}/api/today`).then(todayResp => {
      const allGames = todayResp.data?.games || []
      const filtered = allGames.filter((g: any) => {
        const gameDate = g.game_date || (g.commence_time || g.date || '').slice(0, 10)
        return gameDate === selectedDate
      }).map(normalizeOdds)
      if (filtered.length > 0) {
        setData({ ...todayResp.data, games: filtered, game_count: filtered.length, date: selectedDate })
        setLoading(false)
      } else {
        axios.get(`${API_BASE}/api/games/${selectedDate}`)
          .then(r => { setData(r.data); setLoading(false) })
          .catch(() => { setData(null); setLoading(false) })
      }
    }).catch(() => {
      axios.get(`${API_BASE}/api/games/${selectedDate}`)
        .then(r => { setData(r.data); setLoading(false) })
        .catch(() => { setData(null); setLoading(false) })
    })
  }, [selectedDate])

  useEffect(() => {
    fetchGames()
    // Auto-refresh silently for upcoming dates
    const isUpcoming = selectedDate >= todayStr
    if (isUpcoming) {
      const interval = setInterval(() => fetchGames(true), 60000)
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
    if (direction < 0) {
      const earlier = gameDates.filter(d => d.date < selectedDate)
      if (earlier.length > 0) setSelectedDate(earlier[earlier.length - 1].date)
    } else {
      const later = gameDates.filter(d => d.date > selectedDate)
      if (later.length > 0) setSelectedDate(later[0].date)
    }
  }

  const formatDateLabel = (d: string) => {
    const dt = new Date(d + 'T12:00:00')
    return dt.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
  }

  return (
    <div>
      <div className="date-nav" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <button onClick={() => jumpToNearestGame(-1)} title="Previous date with games">&laquo;</button>
        <CalendarPicker
          selectedDate={selectedDate}
          onSelect={setSelectedDate}
          gameDates={gameDates}
          todayStr={todayStr}
        />
        <button onClick={() => jumpToNearestGame(1)} title="Next date with games">&raquo;</button>
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

    try {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }

      const container = containerRef.current
      container.innerHTML = ''

      const chart = createChart(container, {
        width: container.clientWidth,
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
        color: 'rgb(50, 50, 255)',
        lineWidth: 2,
      })

      // Deduplicate dates (keep last pnl per date), sort ascending
      const dateMap = new Map<string, number>()
      for (const p of data) {
        dateMap.set(p.date, p.pnl)
      }
      const chartData = Array.from(dateMap.entries())
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([date, pnl]) => ({ time: date as any, value: pnl }))

      series.setData(chartData)
      chart.timeScale().fitContent()
    } catch (e) {
      console.warn('PnLChart render error:', e)
    }

    return () => {
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
  const [betType, setBetType] = useState<BetType>('ats')

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
      bet_type: betType,
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
        {filters.map((f, i) => {
          const isSideFilter = ['is_favorite', 'is_underdog', 'is_home', 'is_away', 'is_neutral', 'is_conference'].includes(f.stat)
          return (
          <div key={i} className="filter-row">
            <select value={f.stat} onChange={e => {
              const next = [...filters]
              const newStat = e.target.value
              const isSide = ['is_favorite', 'is_underdog', 'is_home', 'is_away', 'is_neutral', 'is_conference'].includes(newStat)
              if (isSide) {
                next[i] = { ...next[i], stat: newStat, op: '==', value: '1', compare_col: undefined }
              } else {
                next[i] = { ...next[i], stat: newStat }
                const pair = STAT_VS_PAIRS[newStat]
                if (pair && next[i].compare_col) next[i].compare_col = pair
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
            {isSideFilter ? (
              <span className="vs-label" style={{ padding: '0 8px' }}>= Yes</span>
            ) : (
              <>
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
                    <button className="btn btn-sm" title="Compare vs opponent stat" onClick={() => {
                      const pair = STAT_VS_PAIRS[f.stat] || f.stat.replace('team_', 'opp_')
                      const next = [...filters]
                      next[i] = { ...next[i], compare_col: pair, value: '0' }
                      setFilters(next)
                    }}>vs</button>
                  </>
                )}
              </>
            )}
            <button className="btn btn-sm btn-danger" onClick={() => removeFilter(i)}>x</button>
          </div>
          )
        })}
        <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
          <button className="btn btn-sm" onClick={addFilter}>+ Filter</button>
        </div>

        <div style={{ display: 'flex', gap: 4, marginTop: 10 }}>
          <span style={{ fontSize: 10, color: '#657b83', alignSelf: 'center', marginRight: 2 }}>Bet Type:</span>
          <button className={`trends-filter-btn ${betType === 'ats' ? 'active' : ''}`} onClick={() => setBetType('ats')}>ATS</button>
          <button className={`trends-filter-btn ${betType === 'ml' ? 'active' : ''}`} onClick={() => setBetType('ml')}>ML</button>
        </div>
        <div style={{ display: 'flex', gap: 4, marginTop: 8, flexWrap: 'wrap' }}>
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
// TeamPage Component
// ============================================================================

function TeamPage() {
  const [teams, setTeams] = useState<Array<{ name: string; conference: string }>>([])
  const [selectedTeam, setSelectedTeam] = useState('')
  const [query, setQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const [games, setGames] = useState<Game[]>([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    axios.get(`${API_BASE}/api/teams`, { params: { scope: 'all' } }).then(r => {
      const sorted = (r.data as Array<{ name: string; conference: string }>)
        .sort((a, b) => a.name.localeCompare(b.name))
      setTeams(sorted)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selectedTeam) { setGames([]); return }
    setLoading(true)
    axios.get(`${API_BASE}/api/teams/${encodeURIComponent(selectedTeam)}/games`)
      .then(r => { setGames(r.data.games || []); setLoading(false) })
      .catch(() => { setGames([]); setLoading(false) })
  }, [selectedTeam])

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) setShowDropdown(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = query
    ? teams.filter(t => t.name.toLowerCase().includes(query.toLowerCase()) || t.conference.toLowerCase().includes(query.toLowerCase()))
    : teams

  const pickTeam = (name: string) => {
    setSelectedTeam(name)
    setQuery(name)
    setShowDropdown(false)
  }

  return (
    <div>
      <div className="team-search-wrapper" ref={dropdownRef} style={{ marginBottom: '1rem' }}>
        <input
          ref={inputRef}
          className="team-search-input"
          type="text"
          placeholder="Search for a team..."
          value={query}
          onChange={e => { setQuery(e.target.value); setShowDropdown(true) }}
          onFocus={() => setShowDropdown(true)}
        />
        {showDropdown && filtered.length > 0 && (
          <div className="team-search-dropdown">
            {filtered.map(t => (
              <button
                key={t.name}
                className={`team-search-option ${t.name === selectedTeam ? 'active' : ''}`}
                onClick={() => pickTeam(t.name)}
              >
                {t.name} <span className="team-conf">({t.conference})</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="loading">Loading games...</div>
      ) : !selectedTeam ? (
        <div className="no-games">Select a team to view their game log.</div>
      ) : games.length === 0 ? (
        <div className="no-games">No games found for {selectedTeam}.</div>
      ) : (
        <div className="games-grid">
          {games.map(game => (
            <GameCard key={game.game_id} game={game} />
          ))}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// App Component
// ============================================================================

function App() {
  const [view, setView] = useState<View>('scores')
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
          className={`nav-btn ${view === 'scores' ? 'active' : ''}`}
          onClick={() => setView('scores')}
        >
          All Scores
        </button>
        <button
          className={`nav-btn ${view === 'team' ? 'active' : ''}`}
          onClick={() => setView('team')}
        >
          Team
        </button>
        <button
          className={`nav-btn ${view === 'backtest' ? 'active' : ''}`}
          onClick={() => setView('backtest')}
        >
          Backtest
        </button>
        <button
          className={`nav-btn ${view === 'trends' ? 'active' : ''}`}
          onClick={() => setView('trends')}
        >
          Trends
        </button>
        <div className="header-spacer" />
        {updatedAt && (
          <span className="updated-at">
            Updated: {new Date(updatedAt).toLocaleTimeString()}
          </span>
        )}
      </div>
      <div className="main-content">
        {view === 'scores' && <TodayDashboard />}
        {view === 'team' && <TeamPage />}
        {view === 'backtest' && <BacktestPage />}
        {view === 'trends' && <TrendsPage />}
      </div>
    </div>
  )
}

// ============================================================================
// TrendsPage Component
// ============================================================================

interface TrendTimeframe {
  wins: number
  losses: number
  n: number
  win_pct: number
  roi: number
  profit: number
  pnl: { date: string; pnl: number }[]
}

interface TrendStrategy {
  id: string
  side: string
  venue: string
  bucket: string
  label: string
  timeframes: { [key: string]: TrendTimeframe }
}

interface TrendGame {
  date: string
  team: string
  opponent: string
  spread: number
  spread_price: number
  team_score: number
  opp_score: number
  margin: number
  ats_margin: number
  covered: boolean
  stake: number
  profit: number
  side: string
  venue: string
  bucket: string
  phase: string
  conference: string
  ml_price: number | null
  ml_covered: boolean
  ml_stake: number | null
  ml_profit: number | null
}

type BetType = 'ats' | 'ml'

interface TrendsData {
  strategies: TrendStrategy[]
  buckets: string[]
  games: TrendGame[]
  conferences: string[]
  power5: string[]
  conf_tiers: { [tier: string]: string[] }
}

const TIMEFRAME_LABELS: { [key: string]: string } = {
  all: 'All Games',
  regular: 'Regular Season',
  conference_regular: 'Conf. Regular',
  non_conference_regular: 'Non-Conf. Regular',
  conference_tournament: 'Conf. Tournament',
  ncaa_tournament: 'NCAA Tournament',
}

const TIMEFRAME_KEYS = Object.keys(TIMEFRAME_LABELS)

function TrendsChart({ data }: { data: { date: string; pnl: number }[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return

    try {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }

      const container = containerRef.current
      container.innerHTML = ''

      const chart = createChart(container, {
        width: container.clientWidth,
        height: 320,
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
        rightPriceScale: {
          borderColor: '#93a1a1',
        },
        crosshair: {
          horzLine: { labelBackgroundColor: '#586e75' },
          vertLine: { labelBackgroundColor: '#586e75' },
        },
      })
      chartRef.current = chart

      // Determine line color from final P&L
      const finalPnl = data[data.length - 1]?.pnl ?? 0
      const lineColor = finalPnl >= 0 ? 'rgb(50, 50, 255)' : 'rgb(255, 50, 150)'

      const series = chart.addSeries(LineSeries, {
        color: lineColor,
        lineWidth: 2,
      })

      // Zero line
      const zeroSeries = chart.addSeries(LineSeries, {
        color: '#93a1a1',
        lineWidth: 1,
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      })

      const dateMap = new Map<string, number>()
      for (const p of data) {
        dateMap.set(p.date, p.pnl)
      }
      const chartData = Array.from(dateMap.entries())
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([date, pnl]) => ({ time: date as any, value: pnl }))

      series.setData(chartData)

      if (chartData.length >= 2) {
        zeroSeries.setData([
          { time: chartData[0].time, value: 0 },
          { time: chartData[chartData.length - 1].time, value: 0 },
        ])
      }

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
    } catch (e) {
      console.warn('TrendsChart render error:', e)
    }
  }, [data])

  return <div ref={containerRef} className="trends-chart" />
}

function TrendsPage() {
  const [data, setData] = useState<TrendsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string>('favorite_all_all')
  const [timeframe, setTimeframe] = useState<string>('all')
  const [confFilter, setConfFilter] = useState<string>('all')
  const [betType, setBetType] = useState<BetType>('ats')

  useEffect(() => {
    setLoading(true)
    axios.get(`${API_BASE}/api/trends`)
      .then(r => {
        setData(r.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading trends...</div>
  if (!data || !data.games?.length) return <div className="no-games">No trend data available</div>

  const PHASE_MAP: { [tf: string]: string[] | null } = {
    all: null,
    regular: ['conference_regular', 'non_conference_regular'],
    conference_regular: ['conference_regular'],
    non_conference_regular: ['non_conference_regular'],
    conference_tournament: ['conference_tournament'],
    ncaa_tournament: ['ncaa_tournament'],
  }

  const tiers = data.conf_tiers ?? {}
  const tierKeys = ['power5', 'high_major', 'mid_major', 'low_major'] as const
  const tierLabels: { [k: string]: string } = {
    power5: 'Power 5', high_major: 'High Major', mid_major: 'Mid Major', low_major: 'Low Major',
  }
  const allTierConfs = new Set(tierKeys.flatMap(t => tiers[t] ?? []))

  // Apply conference + timeframe filter to all games
  const baseGames = data.games.filter(g => {
    // Conference filter
    if (confFilter === 'power5' || confFilter === 'high_major' || confFilter === 'mid_major' || confFilter === 'low_major') {
      if (!(tiers[confFilter] ?? []).includes(g.conference)) return false
    } else if (confFilter !== 'all') {
      if (g.conference !== confFilter) return false
    }
    // Timeframe filter
    const phases = PHASE_MAP[timeframe]
    if (phases && !phases.includes(g.phase)) return false
    return true
  })

  // Compute stats from a set of games
  const computeStats = (games: TrendGame[]): TrendTimeframe => {
    // For ML, filter to games that have ML data
    const effective = betType === 'ml'
      ? games.filter(g => g.ml_price != null && g.ml_stake != null)
      : games
    if (effective.length === 0) {
      return { wins: 0, losses: 0, n: 0, win_pct: 0, roi: 0, profit: 0, pnl: [] }
    }
    const wins = effective.filter(g =>
      betType === 'ml' ? g.ml_covered : g.covered
    ).length
    const n = effective.length
    const losses = n - wins
    const totalProfit = effective.reduce((s, g) =>
      s + (betType === 'ml' ? (g.ml_profit ?? 0) : g.profit), 0)
    const totalRisked = effective.reduce((s, g) =>
      s + (betType === 'ml' ? (g.ml_stake ?? 0) : g.stake), 0)
    const roi = totalRisked > 0 ? (totalProfit / totalRisked) * 100 : 0

    const sorted = [...effective].sort((a, b) => a.date.localeCompare(b.date))
    let cum = 0
    const pnl = sorted.map(g => {
      cum += betType === 'ml' ? (g.ml_profit ?? 0) : g.profit
      return { date: g.date, pnl: Math.round(cum * 100) / 100 }
    })

    return {
      wins, losses, n,
      win_pct: Math.round((wins / n) * 10000) / 10000,
      roi: Math.round(roi * 100) / 100,
      profit: Math.round(totalProfit * 100) / 100,
      pnl,
    }
  }

  // Compute stats for a strategy key (side/venue/bucket) from baseGames
  const getStats = (side: string, venue: string, bucket: string): TrendTimeframe => {
    let games = baseGames.filter(g => g.side === side)
    if (venue !== 'all') games = games.filter(g => g.venue === venue)
    if (bucket !== 'all') games = games.filter(g => g.bucket === bucket)
    return computeStats(games)
  }

  const selected = data.strategies.find(s => s.id === selectedId)
  const selectedStats = selected ? getStats(selected.side, selected.venue, selected.bucket) : null
  const chartData = selectedStats?.pnl ?? []

  const venues = ['all', 'home', 'away', 'neutral']
  const buckets = ['all', ...data.buckets]

  const venueLabels: { [k: string]: string } = {
    all: 'All', home: 'Home', away: 'Away', neutral: 'Neutral',
  }

  const renderRecord = (tf: TrendTimeframe) => {
    if (tf.n === 0) return <span className="text-muted">--</span>
    return <>{tf.wins}-{tf.losses}</>
  }
  const renderRoi = (tf: TrendTimeframe) => {
    if (tf.n === 0) return <span className="text-muted">--</span>
    const cls = tf.roi > 0 ? 'positive' : tf.roi < 0 ? 'negative' : ''
    return <span className={cls}>{tf.roi > 0 ? '+' : ''}{tf.roi.toFixed(1)}%</span>
  }
  const renderProfit = (tf: TrendTimeframe) => {
    if (tf.n === 0) return <span className="text-muted">--</span>
    const cls = tf.profit > 0 ? 'positive' : tf.profit < 0 ? 'negative' : ''
    return <span className={cls}>{tf.profit > 0 ? '+' : ''}{tf.profit.toFixed(2)}u</span>
  }

  const renderSideSection = (side: 'favorite' | 'underdog') => {
    const btLabel = betType === 'ml' ? 'ML' : 'ATS'
    const sideLabel = side === 'favorite' ? `FAVORITES ${btLabel}` : `UNDERDOGS ${btLabel}`
    return (
      <div key={side} className="trends-section">
        <div className="section-header">{sideLabel}</div>
        {venues.map(venue => (
          <div key={venue} className="trends-venue-group">
            <div className="trends-venue-label">{venueLabels[venue]} Teams</div>
            <table className="trends-table">
              <thead>
                <tr>
                  <th>Spread</th>
                  <th>Record</th>
                  <th>ROI</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {buckets.map(bucket => {
                  const id = `${side}_${venue}_${bucket}`
                  const tf = getStats(side, venue, bucket)
                  const isSelected = id === selectedId
                  return (
                    <tr
                      key={id}
                      className={`trends-row ${isSelected ? 'selected' : ''} ${tf.n === 0 ? 'empty' : ''}`}
                      onClick={() => setSelectedId(id)}
                    >
                      <td className="trends-bucket">{bucket === 'all' ? 'All' : bucket}</td>
                      <td className="trends-record">{renderRecord(tf)}</td>
                      <td className="trends-roi">{renderRoi(tf)}</td>
                      <td className="trends-profit">{renderProfit(tf)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    )
  }

  const selectedLabel = selected?.label ?? 'Favorites / All / All Spreads'
  const confLabel = confFilter === 'all' ? 'All Conferences'
    : tierLabels[confFilter] ?? confFilter

  // Filtered games for the bets table
  const filteredGames = baseGames
    .filter(g => {
      if (!selected) return false
      if (g.side !== selected.side) return false
      if (selected.venue !== 'all' && g.venue !== selected.venue) return false
      if (selected.bucket !== 'all' && g.bucket !== selected.bucket) return false
      return true
    })
    .sort((a, b) => b.date.localeCompare(a.date))

  return (
    <div className="trends-page">
      <div className="trends-chart-section">
        <div className="trends-chart-header">
          <div className="trends-chart-title-row">
            <div className="trends-chart-title">{selectedLabel}</div>
            <div className="trends-bet-toggle">
              <button
                className={`trends-filter-btn ${betType === 'ats' ? 'active' : ''}`}
                onClick={() => setBetType('ats')}
              >ATS</button>
              <button
                className={`trends-filter-btn ${betType === 'ml' ? 'active' : ''}`}
                onClick={() => setBetType('ml')}
              >ML</button>
            </div>
          </div>
          <div className="trends-chart-subtitle">
            {confLabel} &mdash; {TIMEFRAME_LABELS[timeframe]}
            {selectedStats && selectedStats.n > 0 && (
              <> &mdash; {selectedStats.wins}-{selectedStats.losses} ({(selectedStats.win_pct * 100).toFixed(1)}%) &mdash; ROI: {selectedStats.roi > 0 ? '+' : ''}{selectedStats.roi.toFixed(1)}% &mdash; P&L: {selectedStats.profit > 0 ? '+' : ''}{selectedStats.profit.toFixed(2)}u</>
            )}
          </div>
        </div>
        <TrendsChart data={chartData} />
        <div className="trends-filter-bar">
          <div className="trends-filter-row">
            <button
              className={`trends-filter-btn ${confFilter === 'all' ? 'active' : ''}`}
              onClick={() => setConfFilter('all')}
            >All Conferences</button>
          </div>
          {tierKeys.map(tier => (
            <div key={tier} className="trends-filter-row">
              <button
                className={`trends-filter-btn tier-btn ${confFilter === tier ? 'active' : ''}`}
                onClick={() => setConfFilter(tier)}
              >{tierLabels[tier]}</button>
              {(tiers[tier] ?? []).map(c => (
                <button
                  key={c}
                  className={`trends-filter-btn ${confFilter === c ? 'active' : ''}`}
                  onClick={() => setConfFilter(c)}
                >{c}</button>
              ))}
            </div>
          ))}
          <div className="trends-filter-row">
            {TIMEFRAME_KEYS.map(tf => (
              <button
                key={tf}
                className={`trends-filter-btn ${timeframe === tf ? 'active' : ''}`}
                onClick={() => setTimeframe(tf)}
              >
                {TIMEFRAME_LABELS[tf]}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="trends-tables">
        {renderSideSection('favorite')}
        {renderSideSection('underdog')}
      </div>
      {filteredGames.length > 0 && (
        <div className="trends-bets-section">
          <div className="section-header">
            Bet History &mdash; {selectedLabel} &mdash; {confLabel} &mdash; {TIMEFRAME_LABELS[timeframe]} ({filteredGames.length} bets)
          </div>
          <div className="trends-bets-scroll">
            <table className="trends-bets-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Team</th>
                  <th>Opponent</th>
                  <th>Venue</th>
                  <th>Spread</th>
                  <th>Price</th>
                  <th>Score</th>
                  <th>ATS Margin</th>
                  <th>Result</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {filteredGames.map((g, i) => {
                  const won = betType === 'ml' ? g.ml_covered : g.covered
                  const prc = betType === 'ml' ? g.ml_price : g.spread_price
                  const pfl = betType === 'ml' ? (g.ml_profit ?? 0) : g.profit
                  return (
                    <tr key={i} className={won ? 'bet-win' : 'bet-loss'}>
                      <td>{g.date}</td>
                      <td className="bet-team">{g.team}</td>
                      <td>{g.opponent}</td>
                      <td className="bet-venue">{g.venue}</td>
                      <td>{formatSpread(g.spread)}</td>
                      <td>{prc != null ? (prc > 0 ? '+' : '') + prc : '—'}</td>
                      <td>{g.team_score}-{g.opp_score}</td>
                      <td className={g.ats_margin > 0 ? 'positive' : 'negative'}>
                        {g.ats_margin > 0 ? '+' : ''}{g.ats_margin.toFixed(1)}
                      </td>
                      <td className={won ? 'covered' : 'missed'}>
                        {won ? (betType === 'ml' ? 'WIN' : 'COVER') : (betType === 'ml' ? 'LOSS' : 'MISS')}
                      </td>
                      <td className={pfl > 0 ? 'positive' : 'negative'}>
                        {pfl > 0 ? '+' : ''}{pfl.toFixed(2)}u
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
