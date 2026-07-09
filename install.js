// Install: create an isolated venv and install ollamaq from the cloned repo.
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",                 // Pinokio-managed virtualenv in ./env
        message: [
          "python -m pip install --upgrade pip",
          "pip install .",           // installs this repo as the `ollamaq` package
        ],
      },
    },
    {
      method: "notify",
      params: {
        html: "ollamaq installed. Make sure <b>Ollama</b> is running and the models in <code>config/models.yaml</code> are pulled, then click <b>Start daemon</b>.",
      },
    },
  ],
};
