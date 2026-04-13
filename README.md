# Scene Narrator 🎙️

An AI-powered accessibility tool that generates real-time, informative descriptions of images and surroundings for visually impaired users. Scene Narrator combines local object detection with Google's Gemini AI to deliver both fast responses and rich scene understanding.

---

## Features

- **Two-Tier Scene Analysis** — A fast local YOLOv8 model identifies objects in milliseconds, while Gemini 2.5 Flash provides detailed, natural-language scene descriptions.
- **Conversational Q&A** — Users can ask follow-up questions about an image (e.g., *"What color is the car?"* or *"Is anyone wearing a hat?"*) and receive direct, concise answers.
- **Location Awareness** — Integrates with OpenCage Geocoding to describe the user's current location and help find nearby points of interest.
- **Performance Metrics** — Every response includes latency, word count, and reliability stats so you can monitor system health.
- **Accessible Web Interface** — A clean Flask-based UI designed with screen-reader compatibility in mind.

---

## How It Works

```
Camera / Image Upload
        │
        ▼
┌──────────────────┐     ┌──────────────────────┐
│  Tier 1: YOLOv8  │────▶│  Instant object list  │
│  (Local, ~50ms)  │     │  "person, cup, laptop" │
└──────────────────┘     └──────────────────────┘
        │
        ▼
┌──────────────────┐     ┌──────────────────────┐
│  Tier 2: Gemini  │────▶│  Rich narration       │
│  (Cloud AI)      │     │  "A person is sitting  │
└──────────────────┘     │   at a desk with..."   │
                         └──────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Object Detection | Ultralytics YOLOv8 (Nano) |
| Scene Narration | Google Gemini 2.5 Flash |
| Geocoding | OpenCage API |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, CSS, JavaScript |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Google Gemini API key](https://ai.google.dev/)
- An [OpenCage API key](https://opencagedata.com/) (for location features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Trickky337/Scene-Narrator.git
   cd Scene-Narrator
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API keys**

   Open `app.py` and replace the placeholder keys with your own:
   ```python
   os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key'
   OPENCAGE_API_KEY = 'your-opencage-api-key'
   ```
   > **Tip:** For better security, use environment variables instead of hardcoding keys.

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser** and navigate to `http://localhost:5000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `POST` | `/narrate` | Accepts an image and optional question; returns a Gemini-powered scene description |
| `POST` | `/fast_detect` | Accepts an image; returns a fast list of detected objects via YOLOv8 |
| `POST` | `/location_info` | Accepts latitude/longitude (and optional question); returns location details or nearby POIs |

---

## Project Structure

```
Scene-Narrator/
├── app.py               # Flask server & all backend logic
├── requirements.txt     # Python dependencies
├── yolov8n.pt           # YOLOv8 Nano model weights
├── yolov5s.onnx         # YOLOv5 Small model (ONNX format)
├── templates/
│   └── index.html       # Frontend web interface
├── .gitignore
└── README.md
```

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Future Improvements

- Real-time video narration via webcam stream
- Text-to-speech integration for hands-free use
- Support for multiple languages
- Mobile-optimized progressive web app (PWA)
- User preference profiles for description verbosity

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework
- [Google Gemini](https://ai.google.dev/) for multimodal AI capabilities
- [OpenCage](https://opencagedata.com/) for geocoding services

---

## Contact

**Savani Naman** — [@Trickky337](https://github.com/Trickky337)

Project Link: [https://github.com/Trickky337/Scene-Narrator](https://github.com/Trickky337/Scene-Narrator)

---

## License

This project is for educational purposes. See the repository for license details.
