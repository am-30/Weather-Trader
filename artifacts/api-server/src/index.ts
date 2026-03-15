import app, { streamlitProxy } from "./app";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const server = app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

// Forward WebSocket upgrades (required for Streamlit)
server.on("upgrade", streamlitProxy.upgrade);
