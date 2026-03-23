import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { createChart, type IChartApi } from 'lightweight-charts'
import { API_HOST } from '../App'

interface Filter {
  id: number
  stat: string
  compare: string
  value: string
  opp_stat: string
}

interface BacktestResult {
  filters: string[]
  side: string
  n: number
  wins: number
  losses: number
  win_pct: number
  roi: number
  profitable: boolean
  p_value: number
  ci_low: number
  ci_high: number
  curve: { date: string; profit: number; team: string; opponent: string }[]
}

const COMPARE_OPTIONS = [
  { value: 'gt', label: 'Stat > Value' },
  { value: 'lt', label: 'Stat < Value' },
  { value: 'gt_opp', label: 'Stat > Opponent' },
  { value: 'lt_opp', label: 'Stat < Opponent' },
  { value: 'gt_opp_margin', label: 'Stat > Opp by Margin' },
  { value: 'lt_opp_margin', label: 'Stat < Opp by Margin' },
  { value: 'rank_gt', label: 'Team Rank > Pctile' },
  { value: 'rank_lt', label: 'Team Rank < Pctile' },
  { value: 'opp_rank_gt', label: 'Opp Rank > Pctile' },
  { value: 'opp_rank_lt', label: 'Opp Rank < Pctile' },
]

const SIDE_OPTIONS = [
  { value: 'all', label: 'All Games' },
  { value: 'favorites', label: 'Favorites Only' },
  { value: 'underdogs', label: 'Underdogs Only' },
  { value: 'home', label: 'Home Only' },
  { value: 'away', label: 'Away Only' },
]

function needsValue(compare: string): boolean {
  return ['gt', 'lt', 'gt_opp_margin', 'lt_opp_margin', 'rank_gt', 'rank_lt', 'opp_rank_gt', 'opp_rank_lt'].includes(compare)
}

function needsOppStat(compare: string): boolean {
  return ['gt_opp', 'lt_opp', 'gt_opp_margin', 'lt_opp_margin'].includes(compare)
}

function ResultEquityCurve({ curve }: { curve: BacktestResult['curve'] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || curve.length === 0) return

    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 250,
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
      .map(([time, value]) => ({ time, value }))

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
  }, [curve])

  return <div ref={containerRef} className="result-equity-chart" />
}


export default function Backtester() {
  const [stats, setStats] = useState<string[]>([])
  const [filters, setFilters] = useState<Filter[]>([
    { id: 1, stat: 'win_pct', compare: 'gt_opp', value: '', opp_stat: 'win_pct' },
  ])
  const [side, setSide] = useState('all')
  const [minGames, setMinGames] = useState(5)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState(false)
  const nextId = useRef(2)

  useEffect(() => {
    axios.get(`${API_HOST}/api/stats`).then(r => setStats(r.data.stats || [])).catch(() => {})
  }, [])

  const addFilter = () => {
    setFilters([...filters, {
      id: nextId.current++,
      stat: stats[0] || 'win_pct',
      compare: 'gt',
      value: '',
      opp_stat: stats[0] || 'win_pct',
    }])
  }

  const removeFilter = (id: number) => {
    setFilters(filters.filter(f => f.id !== id))
  }

  const updateFilter = (id: number, field: keyof Filter, value: string) => {
    setFilters(filters.map(f => f.id === id ? { ...f, [field]: value } : f))
  }

  const runBacktest = () => {
    setLoading(true)
    const payload = {
      filters: filters.map(f => ({
        stat: f.stat,
        compare: f.compare,
        value: needsValue(f.compare) ? parseFloat(f.value) || 0 : null,
        opp_stat: needsOppStat(f.compare) ? f.opp_stat : null,
      })),
      side,
      min_games: minGames,
    }
    axios.post(`${API_HOST}/api/backtest`, payload)
      .then(r => setResult(r.data))
      .catch(() => setResult(null))
      .finally(() => setLoading(false))
  }

  return (
    <div className="backtest-container">
      {/* Filters panel */}
      <div className="backtest-filters">
        <div className="filter-section-title">Side</div>
        <select className="filter-select" value={side} onChange={e => setSide(e.target.value)}>
          {SIDE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>

        <div className="filter-section-title" style={{ marginTop: 8 }}>Min Games Played</div>
        <input className="filter-input" type="number" value={minGames}
               onChange={e => setMinGames(parseInt(e.target.value) || 0)} />

        <div className="filter-section-title" style={{ marginTop: 8 }}>Conditions</div>

        {filters.map(f => (
          <div key={f.id} style={{ padding: 8, background: 'var(--bg-main)', marginBottom: 4 }}>
            <div className="filter-row">
              <select className="filter-select" value={f.stat}
                      onChange={e => updateFilter(f.id, 'stat', e.target.value)}>
                {stats.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <select className="filter-select" value={f.compare}
                      onChange={e => updateFilter(f.id, 'compare', e.target.value)}>
                {COMPARE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
              <button className="filter-remove-btn" onClick={() => removeFilter(f.id)}>X</button>
            </div>
            <div className="filter-row" style={{ marginTop: 4 }}>
              {needsOppStat(f.compare) && (
                <select className="filter-select" value={f.opp_stat}
                        onChange={e => updateFilter(f.id, 'opp_stat', e.target.value)}>
                  {stats.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              )}
              {needsValue(f.compare) && (
                <input className="filter-input" type="number" step="any" placeholder="value"
                       value={f.value}
                       onChange={e => updateFilter(f.id, 'value', e.target.value)} />
              )}
            </div>
          </div>
        ))}

        <button className="filter-add-btn" onClick={addFilter}>+ Add Condition</button>

        <button className="run-backtest-btn" onClick={runBacktest} disabled={loading}>
          {loading ? 'Running...' : 'Run Backtest'}
        </button>

        {result && result.n > 0 && (
          <div style={{ fontSize: 10, color: 'var(--base00)', marginTop: 4 }}>
            Filters: {result.filters.join(' AND ')}
          </div>
        )}
      </div>

      {/* Results */}
      <div className="backtest-results">
        {result && result.n === 0 && (
          <div className="empty-state">No games matched your filters</div>
        )}

        {result && result.n > 0 && (
          <div className="backtest-result-card">
            <div className="result-stat-grid">
              <div className="result-stat">
                <div className="result-stat-label">Games</div>
                <div className="result-stat-value">{result.n}</div>
              </div>
              <div className="result-stat">
                <div className="result-stat-label">Record</div>
                <div className="result-stat-value">{result.wins}-{result.losses}</div>
              </div>
              <div className="result-stat">
                <div className="result-stat-label">ATS Win%</div>
                <div className={`result-stat-value ${result.win_pct > 0.524 ? 'positive' : result.win_pct < 0.476 ? 'negative' : ''}`}>
                  {(result.win_pct * 100).toFixed(1)}%
                </div>
              </div>
              <div className="result-stat">
                <div className="result-stat-label">ROI</div>
                <div className={`result-stat-value ${result.roi > 0 ? 'positive' : 'negative'}`}>
                  {result.roi > 0 ? '+' : ''}{result.roi.toFixed(1)}%
                </div>
              </div>
              <div className="result-stat">
                <div className="result-stat-label">p-value</div>
                <div className={`result-stat-value ${result.p_value < 0.05 ? 'positive' : ''}`}>
                  {result.p_value.toFixed(4)}
                </div>
              </div>
            </div>

            <div style={{ fontSize: 11, marginBottom: 8, color: 'var(--base00)' }}>
              95% CI: {(result.ci_low * 100).toFixed(1)}% - {(result.ci_high * 100).toFixed(1)}%
              {result.profitable && <span className="positive" style={{ marginLeft: 8 }}>PROFITABLE ({'>'} 52.4%)</span>}
            </div>

            <ResultEquityCurve curve={result.curve} />
          </div>
        )}

        {!result && !loading && (
          <div className="empty-state">
            Configure filters and click "Run Backtest" to see results.<br /><br />
            <span style={{ fontSize: 11 }}>
              Available comparisons:<br />
              - Stat &gt; Value: filter by absolute stat value<br />
              - Stat &gt; Opponent: team stat better than opponent<br />
              - Stat &gt; Opp by Margin: team stat better by X<br />
              - Team/Opp Rank: filter by percentile ranking (0-100)<br />
              &nbsp;&nbsp;Example: Team ppg rank &gt; 75 = top 25% scoring
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
