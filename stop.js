// Stop the running daemon (terminates the start.js process).
module.exports = {
  run: [
    { method: "shell.stop", params: { path: "start.js" } },
    { method: "notify", params: { html: "Daemon stopped." } },
  ],
};
