# Raspberry Pi CCTV AI System

An AI-powered CCTV system built with FastAPI, designed to run on Raspberry Pi and accessible over the internet using Cloudflare Tunnel.

## 🧠 Turn Your CCTV DVR/NVR into a Customizable AI with Raspberry Pi 4B!

**💡 No Torch? No Problem!**  
If you're using a **Raspberry Pi 4B (2GB RAM)** and found that **PyTorch isn't supported**, don't worry —  
you can still unlock the power of AI using **ONNX Runtime**.

✅ Supports running **object detection** models like **YOLO**, **CNN**, **CRNN**, etc., directly on the Pi.  
✅ Inference with `.onnx` models runs **locally**, with no cloud dependency required.  
✅ Designed for lightweight, real-time AI on CCTV streams.  

## 🚀 Features

- Real-time CCTV feed processing
- AI video analysis capabilities
- Accessible remotely via secure Cloudflare Tunnel
- Designed for low-power Raspberry Pi environments
- REST API via FastAPI

---

## 📦 Requirements

- Raspberry Pi 4 (or newer)
- Python 3.7+
- onnxruntime
- Virtual environment (recommended)
- FastAPI
- uvicorn
- Cloudflare Tunnel (`cloudflared`)
- Camera module or RTSP stream

---

## 🔧 Installation

1. **Clone this repository**

```bash
git clone https://github.com/yourusername/raspberrypi-cctv-ai.git
cd raspberrypi-cctv-ai
```

2. **Create a virtual environment**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the FastAPI app**

```bash
python main.py
```

App will start at:  
**http://localhost:8000**  
Docs: **http://localhost:8000/docs**

---

## 🌍 Access Over Internet Using Cloudflare Tunnel

### Option 1: (✅ Recommended) Create a persistent, named tunnel

1. **Install cloudflared**

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared-linux-arm64.deb
```

2. **Authenticate with Cloudflare**

```bash
cloudflared login
```

Follow the link in the terminal to authenticate and select your domain.

3. **Create and run a named tunnel**

```bash
cloudflared tunnel create cctv-tunnel
cloudflared tunnel route dns cctv-tunnel cctv.example.com
```

4. **Configure tunnel service**

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: cctv-tunnel
credentials-file: /home/ai/.cloudflared/cctv-tunnel.json

ingress:
  - hostname: cctv.example.com
    service: http://localhost:8000
  - service: http_status:404
```

5. **Run the tunnel**

```bash
cloudflared tunnel run cctv-tunnel
```

---

### Option 2: (⚠️ Temporary, not persistent)

Use this for quick testing without authentication.

```bash
cloudflared tunnel --url http://localhost:8000
```

This will generate a temporary `trycloudflare.com` URL (e.g., `https://your-name.trycloudflare.com`) that you can access from anywhere. Keep the terminal open as it will shut down when closed.

---

## 🛠️ Set Up FastAPI as a systemd Service

```bash
sudo nano /etc/systemd/system/fastapi-cctv.service
```

Paste this:

```ini
[Unit]
Description=FastAPI CCTV App
After=network.target

[Service]
User=ai
WorkingDirectory=/home/ai/raspberrypi-cctv-ai
ExecStart=/home/ai/myenv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reexec
sudo systemctl enable fastapi-cctv
sudo systemctl start fastapi-cctv
```

---

## 📁 Project Structure

```
raspberrypi-cctv-ai/
├── app/
├── recordings/
├── templates/
├── main.py
├── requirements.txt
└── README.md
```

---

## 📸 Author

- 🧑‍💻 Built by Yarramsetti Naresh
- 🔗 [LinkedIn](https://www.linkedin.com/in/yarramsetti-naresh/)

---

## 📄 License

This project is licensed under the MIT License.