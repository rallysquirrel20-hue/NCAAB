const path = require('path');

module.exports = {
  apps: [
    {
      name: "ncaab-frontend",
      cwd: path.join(__dirname, "app", "frontend"),
      script: path.join(__dirname, "app", "frontend", "node_modules", "vite", "bin", "vite.js"),
      args: "dev",
      env: {
        NODE_ENV: "development",
      },
    },
    {
      name: "ncaab-backend",
      cwd: path.join(__dirname, "app", "backend"),
      interpreter: path.join(__dirname, "app", "backend", "venv", "Scripts", "python.exe"),
      script: "main.py",
      env: {
        PYTHONPATH: path.join(__dirname, "app", "backend"),
      },
    },
    {
      name: "ncaab-odds",
      cwd: __dirname,
      interpreter: path.join(__dirname, "app", "backend", "venv", "Scripts", "python.exe"),
      script: "ncaab_odds_refresher.py",
      args: "--loop --interval 900",
    },
    {
      name: "ncaab-schedule",
      cwd: __dirname,
      interpreter: path.join(__dirname, "app", "backend", "venv", "Scripts", "python.exe"),
      script: "ncaab_schedule_refresher.py",
      args: "--loop --interval 300",
    },
    {
      name: "ncaab-daily",
      cwd: __dirname,
      interpreter: path.join(__dirname, "app", "backend", "venv", "Scripts", "python.exe"),
      script: "ncaab_daily_builder.py",
      args: "--loop --interval 21600",
    },
  ],
};
