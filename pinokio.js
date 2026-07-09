// Pinokio 1-click launcher for ollamaq.
//
// Pinokio clones this repo, then reads this file to render the app menu. The
// menu is state-aware: before install it offers Install; once the venv exists
// it offers Start/Update/Reset; while the daemon runs it offers Stop.
//
// ollamaq is a CLI harness, not a web app — "Start" runs the queue daemon as a
// persistent terminal process. It needs a local Ollama server (with the models
// from config/models.yaml pulled) reachable at http://localhost:11434.

module.exports = {
  version: "3.0",
  title: "ollamaq",
  description: "Local Ollama offload harness for Claude Code: map-reduce batch jobs + queue daemon.",
  menu: async (kernel, info) => {
    const installed = info.exists("env");           // venv created by install.js?
    const running = info.running("start.js");        // daemon process alive?
    if (running) {
      return [
        { icon: "fa-solid fa-desktop", text: "Daemon (running)", href: "start.js" },
        { icon: "fa-solid fa-stop", text: "Stop daemon", href: "stop.js" },
      ];
    }
    if (installed) {
      return [
        { icon: "fa-solid fa-play", text: "Start daemon", href: "start.js" },
        { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
        { icon: "fa-solid fa-eraser", text: "Reset (delete venv)", href: "reset.js" },
      ];
    }
    return [
      { icon: "fa-solid fa-download", text: "Install", href: "install.js" },
    ];
  },
};
