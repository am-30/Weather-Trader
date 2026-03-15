import express, { type Express } from "express";
import cors from "cors";
import { createProxyMiddleware } from "http-proxy-middleware";
import router from "./routes";

const app: Express = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

// Proxy everything else (HTTP + WS) to the Streamlit app on port 5000
export const streamlitProxy = createProxyMiddleware({
  target: "http://localhost:5000",
  changeOrigin: true,
  ws: true,
});

app.use("/", streamlitProxy);

export default app;
