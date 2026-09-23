// 临时静态服务：只用来把桌面端原型渲染出来截图，用完即删
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const PORT = 8790;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

http.createServer((req, res) => {
  const name = decodeURIComponent((req.url || "/").split("?")[0]);
  const file = path.join(ROOT, name === "/" ? "desktop.html" : name.replace(/^\/+/, ""));
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(data);
  });
}).listen(PORT, "127.0.0.1", () => console.log("http://127.0.0.1:" + PORT));
