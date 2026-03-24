const path = require('path')
const NCAAB_DIR = path.resolve(__dirname, '..')
const PYTHON = path.join(__dirname, 'backend', 'venv', 'Scripts', 'python.exe')

module.exports = {
  apps: [
    {
      name: 'ncaab-backend',
      cwd: path.join(__dirname, 'backend'),
      script: PYTHON,
      args: 'main.py',
      watch: false,
      autorestart: true,
    },
    {
      name: 'ncaab-frontend',
      cwd: path.join(__dirname, 'frontend'),
      script: './node_modules/vite/bin/vite.js',
      args: '--host 0.0.0.0 --port 5174',
      interpreter: 'node',
      watch: false,
      autorestart: true,
    },
    {
      name: 'ncaab-schedule',
      cwd: NCAAB_DIR,
      script: PYTHON,
      args: 'ncaab_schedule_refresher.py --loop --interval 60',
      watch: false,
      autorestart: true,
    },
    {
      name: 'ncaab-odds',
      cwd: NCAAB_DIR,
      script: PYTHON,
      args: 'ncaab_odds_refresher.py --loop --interval 900',
      watch: false,
      autorestart: true,
    },
  ],
}
