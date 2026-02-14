# AutoTinder 🤖❤️

*Because manually swiping at 2 AM is so 2019.*

An AI-powered Android automation tool that analyzes Tinder profiles and swipes for you using Google's Gemini AI. Born from pure 2 AM motivation and questionable life choices.

## What Does This Thing Do?

AutoTinder is your personal dating app assistant that:

- 📸 **Captures & stitches** complete profile screenshots
- 🧠 **AI-powered analysis** using Gemini to evaluate profiles against your criteria
- 👆 **Auto-swipes** based on AI decisions (no more thumb fatigue!)
- 🚫 **Dismisses pop-ups** and detects fake/advertisement profiles
- 📊 **Tracks stats** because who doesn't love metrics on their dating life?

## Tech Stack

**Backend Magic:**
- Python 3.x + Flask for the web interface
- `uiautomator2` for Android device control via ADB
- OpenCV for button detection (template matching FTW)
- Google Gemini AI for the actual thinking
- SQLite for storing your digital dating history

**Frontend:**
- Vanilla JS (because not everything needs React)
- Socket.IO for real-time console logs
- HTML/CSS that doesn't look terrible

## Quick Start

### 1. Prerequisites

You'll need:
- Android device (real or emulator) with developer mode enabled
- Python 3.8+
- ADB installed and working (`adb devices` should list your phone)
- [Gemini API key](https://aistudio.google.com/apikey) (it's free-ish)

### 2. Installation

```bash
git clone https://github.com/yourusername/AutoTinder.git
cd AutoTinder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r autotinder-web/requirements.txt
```

### 3. Configuration

Edit `config.py`:

```python
GEMINI_API_KEY = "your-api-key-here"
DECISION_CRITERIA = "Age 25-35, outdoorsy, good sense of humor"
PACKAGE_NAME = "com.tinder"  # Or whatever dating app you're using
```

### 4. Run It

**Web Interface (recommended):**
```bash
cd autotinder-web
python app.py
```
Open `http://localhost:5000` and watch the magic happen.

**Command Line (for the purists):**
```bash
python automater.py
```

## How It Works

1. **Screenshot spree**: Scrolls through a profile and captures multiple screenshots
2. **Image stitching**: Combines them into one complete profile view
3. **AI consultation**: Sends to Gemini with your criteria ("Does this person seem cool?")
4. **Decision time**: Gemini returns LIKE or DISLIKE with reasoning
5. **Execute**: Taps the appropriate button
6. **Rinse & repeat**: Logs everything and moves to the next profile

### Smart Overlay Detection

The AI distinguishes between:
- ❌ **Blocking overlays** (modals, permissions) → Dismissed automatically
- ✅ **Non-blocking banners** (tooltips) → Ignored
- 🚫 **Fake profiles** (advertisements) → Instant left swipe

## Configuration Deep Dive

Key settings in `config.py`:

```python
MODEL_HEAVY = "gemini-2.5-flash"  # For complex analysis
MODEL_FAST = "gemini-2.5-flash"   # For quick tasks

SHORT_WAIT = 1.0   # Time between actions
MEDIUM_WAIT = 1.5  # Be patient with the UI
LONG_WAIT = 3.0    # Good things take time
```

**Pro tip**: Adjust wait times based on your device speed. Too fast = broken automation, too slow = watching paint dry.

## Features

### Web Dashboard
- 📺 **Live screen mirroring** of your device
- 📝 **Real-time console** with Socket.IO streaming
- ⚙️ **Settings management** via slick web UI
- 📈 **Stats dashboard** (total profiles, like rate, etc.)

### Automation Engine
- 🎯 **Template matching** with OpenCV for pixel-perfect button detection
- 🧩 **Multi-scale detection** (0.3x to 1.1x) because not all screens are created equal
- 🗄️ **Profile logging** to SQLite with timestamps and decisions
- 💾 **Screenshot storage** in `analyzed_profiles/` directory

## File Structure

```
AutoTinder/
├── automater.py              # The brain
├── config.py                 # Your preferences
├── like.png, dislike.png     # Button templates
├── autotinder-web/
│   ├── app.py               # Flask server
│   ├── static/              # JS/CSS
│   ├── templates/           # HTML
│   └── instance/            # SQLite DB
└── analyzed_profiles/       # Your swiping history
```

## Security Notes 🔒

- **Never commit your API key** (seriously, use `.gitignore`)
- All data stays **local** except API calls to Gemini
- Profile screenshots are saved locally (manage disk space!)
- The usual ⚠️ about keeping secrets secret applies

## Troubleshooting

**Device not connecting?**
```bash
adb kill-server
adb start-server
adb devices
```

**Buttons not detected?**
- Check that `like.png`, `dislike.png`, `scroll.png` exist
- Verify device resolution isn't doing weird things
- Try adjusting confidence threshold in code

**AI acting weird?**
- Double-check your `DECISION_CRITERIA` isn't contradictory
- Verify API key is valid and has quota remaining
- Check internet connection (AI can't read minds... yet)

## Performance Tips

- Use `gemini-2.5-flash` for speed (already default)
- Lower wait times for faster execution (may break things)
- Reduce screenshot count for quicker analysis
- Pro mode: Run on emulator for maximum speed

## Limitations

- Built for Tinder specifically (other apps = YMMV)
- Requires Android device or emulator
- Gemini API costs money after free tier
- English profiles work best
- Can't guarantee you'll actually get matches 😅

## Legal Stuff

⚠️ **Disclaimer**: This is for educational/personal use. Don't be evil:
- Respect Tinder's ToS
- Don't use for spam or harassment
- Be a decent human
- Authors not responsible if you get banned

Use at your own risk. Automate responsibly. Touch grass occasionally.

---

**Questions? Issues? Want to contribute?** Open an issue on GitHub!

*P.S. - If this actually helps you find love, I demand wedding invite invitations.*