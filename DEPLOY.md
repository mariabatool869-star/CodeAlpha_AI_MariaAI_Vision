# Streamlit Cloud Deployment

## Required files (included in this repo)

| File | Purpose |
|------|---------|
| `app.py` | Main entry point |
| `requirements.txt` | Python deps (`opencv-python-headless`, CPU PyTorch) |
| `.python-version` | Pins Python **3.11** |
| `packages.txt` | Linux system libraries for OpenCV |
| `.streamlit/config.toml` | Headless server settings |

## Deploy steps

1. Push this folder to GitHub (see commands below).
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → select your repo → set main file to `app.py`.
4. Click **Deploy** and wait 5–15 minutes for the first build.

## Push to GitHub

```bash
cd c:\Users\Maria\Desktop\MariaAI-Vision
git init
git add .
git commit -m "Fix Streamlit Cloud deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

If the app already exists on Streamlit Cloud, push to `main` and click **Reboot app**.

## Cloud limitations

- **No webcam** — use **Upload Video** only.
- **CPU only** — use `yolov8n.pt` for best performance.
- **First Start** — models download on first run (can take 1–2 minutes).

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ImportError: cv2` | Ensure `.python-version`, `packages.txt`, and `opencv-python-headless` are committed |
| App won't start | Check logs in **Manage app** on Streamlit Cloud |
| Out of memory | Use `yolov8n.pt` only; avoid large models on cloud |
