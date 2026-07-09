// Update: pull the latest repo changes and reinstall the package.
module.exports = {
  run: [
    { method: "shell.run", params: { message: "git pull" } },
    {
      method: "shell.run",
      params: { venv: "env", message: "pip install --upgrade ." },
    },
    { method: "notify", params: { html: "ollamaq updated." } },
  ],
};
