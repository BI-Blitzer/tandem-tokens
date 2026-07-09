// Reset: delete the venv so Install can rebuild it from scratch.
// Leaves queue/ and results/ untouched.
module.exports = {
  run: [
    { method: "fs.rm", params: { path: "env" } },
    { method: "notify", params: { html: "venv removed. Click Install to rebuild." } },
  ],
};
