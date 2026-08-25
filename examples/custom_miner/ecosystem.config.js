// pm2 config for the miner. The BROWSERS ARE NOT MANAGED HERE — you start them
// yourself and sign in by hand, which is the whole reason sign-in works (a
// browser a driver launched is refused by some providers). pm2 supervises only
// the miner, and the miner attaches to whatever browsers are up.
//
//   pm2 start ecosystem.config.js
//   pm2 logs hone-miner
//   pm2 save && pm2 startup      # survive a reboot
//
// A restart is safe: the miner closes the tabs it opened, and on an unclean
// kill the next start reclaims them, so tab count stays flat across restarts.
module.exports = {
  apps: [
    {
      name: "hone-miner",
      // Run from the repo root so `.env` and the `rlvr` package are found.
      cwd: "../..",
      script: ".venv/bin/python",
      args: "examples/custom_miner/run_miner.py",
      interpreter: "none",          // script IS the interpreter
      autorestart: true,
      // A crash loop against an unreachable browser should back off, not spin.
      restart_delay: 10000,
      max_restarts: 50,
      min_uptime: "60s",
      // The miner is long-lived and holds Playwright connections; let it grow
      // a little before pm2 decides to recycle it.
      max_memory_restart: "1G",
      // SIGINT lets the miner close its tabs cleanly; give it time to finish.
      kill_timeout: 20000,
      env: {
        PYTHONUNBUFFERED: "1",      // so `pm2 logs` is live, not block-buffered
      },
      out_file: "logs/hone-miner.out.log",
      error_file: "logs/hone-miner.err.log",
      merge_logs: true,
      time: true,
    },
  ],
};
