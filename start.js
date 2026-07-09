// Start the queue daemon as a persistent process.
//
// The daemon runs in the foreground and drains ./queue into ./results. The `on`
// matcher watches for its startup line and releases the Pinokio spinner while
// the process keeps running — submit jobs with `ollamaq submit` from a shell in
// this folder (or the CLI on the host).
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "ollamaq daemon",
        on: [{ event: "/daemon started/", done: true }],
      },
    },
  ],
};
