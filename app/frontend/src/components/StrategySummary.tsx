import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import { createChart, type IChartApi } from 'lightweight-charts'
import { API_HOST } from '../App'

interface StrategyFilter {
  side: string
  venue: string
  spread_lo: number
  spread_hi: number
}

interface StrategyStats {
  name: string
  n: number
  wins: number
  losses: number
  win_pct: number
  roi: number
  units: number
  profitable: boolean
  filter: StrategyFilter
}

interface TimeframeData {
  all_games: StrategyStats[]
  home_away: StrategyStats[]
}

interface GameRow {
  date: string
  team: string
  opponent: string
  home_away: string
  team_score: number | null
  opp_score: number | null
  spread: number | null
  price: number
  covered: boolean
  ats_margin: number | null
  pnl: number
  team_win_pct: number | null
  opp_win_pct: number | null
}

const TIMEFRAME_LABELS: Record<string, string> = {
  full_season: 'Full Season',
  regular_season: 'Regular Season',
  conf_tournaments: 'Conf Tournaments',
  ncaa_tournament: 'NCAA Tournament',
  first_four: 'First Four',
  round_of_64: 'Round of 64',
  round_of_32: 'Round of 32',
  sweet_16: 'Sweet 16',
  elite_8: 'Elite 8',
  final_four: 'Final Four',
}

const DEFAULT_FILTER: StrategyFilter = { side: 'favorite', venue: 'all', spread_lo: -999, spread_hi: 999 }

type SortKey = 'date' | 'team' | 'opponent' | 'spread' | 'price' | 'ats_margin' | 'pnl' | 'team_score'
type SortDir = 'asc' | 'desc'

// ── Equity curve chart ────────────────────────────────────────────────────
function EquityCurveChart({ filter, timeframe, label }: {
  filter: StrategyFilter; timeframe: string; label: string
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    axios.get(`${API_HOST}/api/strategy/equity-curve`, {
      params: {
        side: filter.side,
        venue: filter.venue,
        spread_lo: filter.spread_lo,
        spread_hi: filter.spread_hi,
        timeframe,
      },
    })
      .then(r => {
        const curve: { date: string; profit: number }[] = r.data.curve || []
        if (!containerRef.current) return

        if (chartRef.current) {
          chartRef.current.remove()
          chartRef.current = null
        }

        if (curve.length === 0) return

        const chart = createChart(containerRef.current, {
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
          layout: { background: { color: '#eee8d5' }, textColor: '#586e75', fontSize: 10,
                    fontFamily: "'Fira Code', 'Cascadia Code', Consolas, monospace" },
          grid: { vertLines: { visible: false }, horzLines: { color: '#93a1a1', style: 2 } },
          rightPriceScale: { borderVisible: false },
          crosshair: { mode: 0 },
          timeScale: { borderVisible: false },
        })
        chartRef.current = chart

        const dailyMap = new Map<string, number>()
        for (const p of curve) dailyMap.set(p.date, p.profit)

        const data = Array.from(dailyMap.entries())
          .sort((a, b) => a[0].localeCompare(b[0]))
          .map(([date, value]) => ({ time: date, value }))

        const isPositive = data.length > 0 && data[data.length - 1].value >= 0
        const series = chart.addLineSeries({
          color: isPositive ? '#2aa198' : '#dc322f',
          lineWidth: 2,
          priceLineVisible: false,
        })
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        series.setData(data as any)
        series.createPriceLine({ price: 0, color: '#586e75', lineWidth: 1, lineStyle: 2 })
        chart.timeScale().fitContent()

        const ro = new ResizeObserver(() => {
          if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth })
        })
        ro.observe(containerRef.current)

        return () => {
          ro.disconnect()
          chart.remove()
          chartRef.current = null
        }
      })
      .catch(() => {})

    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [filter, timeframe, label])

  return (
    <div className="equity-curve-panel" style={{ gridColumn: '1 / -1' }}>
      <div className="equity-curve-title">{label}</div>
      <div ref={containerRef} className="equity-chart-wrap" />
    </div>
  )
}

// ── Games drill-down table ────────────────────────────────────────────────
function GamesList({ filter, timeframe, strategyName, onClose }: {
  filter: StrategyFilter; timeframe: string; strategyName: string; onClose: () => void
}) {
  const [games, setGames] = useState<GameRow[]>([])
  const [loading, setLoading] = useState(true)
  const [sortKey, setSortKey] = useState<SortKey>('date')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  useEffect(() => {
    setLoading(true)
    axios.get(`${API_HOST}/api/strategy/games`, {
      params: {
        side: filter.side,
        venue: filter.venue,
        spread_lo: filter.spread_lo,
        spread_hi: filter.spread_hi,
        timeframe,
      },
    })
      .then(r => setGames(r.data.games || []))
      .catch(() => setGames([]))
      .finally(() => setLoading(false))
  }, [filter, timeframe])

  const toggleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir(key === 'date' ? 'desc' : 'desc')
    }
  }, [sortKey])

  const sorted = [...games].sort((a, b) => {
    const av = a[sortKey]
    const bv = b[sortKey]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number)
    return sortDir === 'asc' ? cmp : -cmp
  })

  const cumPnl = (() => {
    const byDate = [...games].sort((a, b) => a.date.localeCompare(b.date))
    let cum = 0
    const map = new Map<string, number>()
    for (const g of byDate) {
      cum += g.pnl
      map.set(`${g.date}_${g.team}_${g.opponent}`, Math.round(cum * 1000) / 1000)
    }
    return map
  })()

  const arrow = (key: SortKey) => sortKey === key ? (sortDir === 'asc' ? ' ^' : ' v') : ''

  if (loading) return <div className="loading">Loading games...</div>

  return (
    <div className="strategy-table-container" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div className="strategy-table-title" style={{ marginBottom: 0 }}>
          {strategyName} — {games.length} games ({TIMEFRAME_LABELS[timeframe]})
        </div>
        <button className="filter-remove-btn" onClick={onClose} style={{ fontSize: 12, padding: '4px 10px' }}>Close</button>
      </div>
      <div style={{ maxHeight: 500, overflowY: 'auto' }}>
        <table className="strategy-table">
          <thead>
            <tr>
              <th onClick={() => toggleSort('date')} style={{ cursor: 'pointer' }}>Date{arrow('date')}</th>
              <th onClick={() => toggleSort('team')} style={{ cursor: 'pointer' }}>Favorite{arrow('team')}</th>
              <th onClick={() => toggleSort('opponent')} style={{ cursor: 'pointer' }}>Underdog{arrow('opponent')}</th>
              <th>Score</th>
              <th onClick={() => toggleSort('spread')} style={{ cursor: 'pointer' }}>Spread{arrow('spread')}</th>
              <th onClick={() => toggleSort('price')} style={{ cursor: 'pointer' }}>Price{arrow('price')}</th>
              <th onClick={() => toggleSort('ats_margin')} style={{ cursor: 'pointer' }}>ATS Margin{arrow('ats_margin')}</th>
              <th>Result</th>
              <th onClick={() => toggleSort('pnl')} style={{ cursor: 'pointer' }}>P&L{arrow('pnl')}</th>
              <th>Cum P&L</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((g, i) => {
              const key = `${g.date}_${g.team}_${g.opponent}`
              const cum = cumPnl.get(key) ?? 0
              return (
                <tr key={i}>
                  <td>{g.date}</td>
                  <td>{g.team}</td>
                  <td>{g.opponent}</td>
                  <td>{g.team_score != null ? `${g.team_score}-${g.opp_score}` : '-'}</td>
                  <td>{g.spread != null ? (g.spread > 0 ? `+${g.spread}` : g.spread) : '-'}</td>
                  <td>{g.price > 0 ? `+${g.price}` : g.price}</td>
                  <td className={g.ats_margin != null ? (g.ats_margin > 0 ? 'positive' : 'negative') : ''}>
                    {g.ats_margin != null ? (g.ats_margin > 0 ? '+' : '') + g.ats_margin.toFixed(1) : '-'}
                  </td>
                  <td className={g.covered ? 'positive' : 'negative'}>
                    {g.covered ? 'W' : 'L'}
                  </td>
                  <td className={g.pnl > 0 ? 'positive' : 'negative'}>
                    {g.pnl > 0 ? '+' : ''}{g.pnl.toFixed(3)}
                  </td>
                  <td className={cum > 0 ? 'positive' : cum < 0 ? 'negative' : ''}>
                    {cum > 0 ? '+' : ''}{cum.toFixed(3)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Strategy row ──────────────────────────────────────────────────────────
function StrategyRow({ s, indent = 0, selected, onClick }: {
  s: StrategyStats; indent?: number; selected: boolean; onClick: () => void
}) {
  if (s.n === 0) return null
  return (
    <tr onClick={onClick} style={{ cursor: 'pointer', background: selected ? 'var(--bg-main)' : undefined }}>
      <td style={{ paddingLeft: 8 + indent * 16 }}>
        {indent > 0 && <span style={{ color: 'var(--base0)', marginRight: 4 }}>{'>'}</span>}
        {s.name}
      </td>
      <td>{s.n}</td>
      <td>{s.wins}-{s.losses}</td>
      <td className={s.win_pct > 0.524 ? 'positive' : s.win_pct < 0.476 ? 'negative' : ''}>
        {(s.win_pct * 100).toFixed(1)}%
      </td>
      <td className={s.units > 0 ? 'positive' : s.units < 0 ? 'negative' : ''}>
        {s.units > 0 ? '+' : ''}{s.units.toFixed(2)}u
      </td>
      <td className={s.roi > 0 ? 'positive' : 'negative'}>
        {s.roi > 0 ? '+' : ''}{s.roi.toFixed(1)}%
      </td>
      <td>{s.profitable ? '+' : ''}</td>
    </tr>
  )
}

// ── Strategy section ──────────────────────────────────────────────────────
function StrategySection({ rows, title, timeframe, selectedName, onSelect }: {
  rows: StrategyStats[]
  title: string
  timeframe: string
  selectedName: string | null
  onSelect: (s: StrategyStats | null) => void
}) {
  if (!rows || rows.length === 0) return null

  const groups: { total: StrategyStats; buckets: StrategyStats[] }[] = []
  for (let i = 0; i < rows.length; i += 5) {
    groups.push({
      total: rows[i],
      buckets: rows.slice(i + 1, i + 5),
    })
  }

  const flatRows: { s: StrategyStats; indent: number }[] = []
  for (const g of groups) {
    flatRows.push({ s: g.total, indent: 0 })
    for (const b of g.buckets) flatRows.push({ s: b, indent: 1 })
  }

  const selectedIdx = selectedName != null
    ? flatRows.findIndex(r => r.s.name === selectedName)
    : -1
  const selected = selectedIdx >= 0 ? flatRows[selectedIdx] : null

  return (
    <>
      <div className="strategy-table-container" style={{ marginBottom: selected ? 0 : 12 }}>
        <div className="strategy-table-title">{title}</div>
        <table className="strategy-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Games</th>
              <th>Record</th>
              <th>ATS Win%</th>
              <th>Units</th>
              <th>ROI</th>
              <th>+</th>
            </tr>
          </thead>
          <tbody>
            {flatRows.map((item, idx) => (
              <StrategyRow
                key={idx}
                s={item.s}
                indent={item.indent}
                selected={selectedIdx === idx}
                onClick={() => onSelect(selectedIdx === idx ? null : item.s)}
              />
            ))}
          </tbody>
        </table>
      </div>
      {selected && selected.s.n > 0 && selected.s.filter && (
        <GamesList
          filter={selected.s.filter}
          timeframe={timeframe}
          strategyName={selected.s.name}
          onClose={() => onSelect(null)}
        />
      )}
    </>
  )
}

// ── Main component ────────────────────────────────────────────────────────
export default function StrategySummary() {
  const [summary, setSummary] = useState<Record<string, TimeframeData> | null>(null)
  const [selectedTf, setSelectedTf] = useState('full_season')
  const [loading, setLoading] = useState(true)
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyStats | null>(null)

  useEffect(() => {
    setLoading(true)
    axios.get(`${API_HOST}/api/strategy/summary`)
      .then(r => setSummary(r.data))
      .catch(() => setSummary(null))
      .finally(() => setLoading(false))
  }, [])

  // Reset selection when timeframe changes
  useEffect(() => { setSelectedStrategy(null) }, [selectedTf])

  if (loading) return <div className="loading">Loading strategy data...</div>
  if (!summary) return <div className="empty-state">Failed to load strategy data</div>

  const tfData = summary[selectedTf]
  const curveFilter = selectedStrategy?.filter ?? DEFAULT_FILTER
  const curveLabel = selectedStrategy
    ? `${selectedStrategy.name} — Equity Curve`
    : 'All Favorites — Equity Curve'

  return (
    <div className="strategy-container">
      <div className="equity-curves-row" style={{ gridTemplateColumns: '1fr' }}>
        <EquityCurveChart
          filter={curveFilter}
          timeframe={selectedTf}
          label={curveLabel}
        />
      </div>

      <div className="timeframe-pills">
        {Object.entries(TIMEFRAME_LABELS).map(([key, label]) => (
          <button key={key} className={`tf-pill ${selectedTf === key ? 'active' : ''}`}
                  onClick={() => setSelectedTf(key)}>
            {label}
          </button>
        ))}
      </div>

      {tfData && (
        <>
          <StrategySection
            title={`${TIMEFRAME_LABELS[selectedTf]} — All Games`}
            rows={tfData.all_games}
            timeframe={selectedTf}
            selectedName={selectedStrategy?.name ?? null}
            onSelect={setSelectedStrategy}
          />
          <StrategySection
            title={`${TIMEFRAME_LABELS[selectedTf]} — Home / Away / Neutral Splits`}
            rows={tfData.home_away}
            timeframe={selectedTf}
            selectedName={selectedStrategy?.name ?? null}
            onSelect={setSelectedStrategy}
          />
        </>
      )}
    </div>
  )
}
