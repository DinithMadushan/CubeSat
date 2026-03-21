# CubeSat Attitude and Orbital Control Simulator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-latest-orange)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview
A realistic **CubeSat attitude and orbital control simulator** built with Pygame, NumPy, and PID controllers. Features interactive 3D visualization, orbital mechanics simulation around multiple planets (Earth, Mars, Jupiter, etc.), live telemetry, attitude graphs, and PID-based stabilization.

**Key Physics:**
- Gravitational orbital dynamics (Newtonian gravity)
- PID attitude control (roll, pitch, yaw)
- Orbital thrust correction for stable circular orbits

![Demo Screenshot](screenshots/demo.gif)
*(Add your screenshot/video here)*

## ✨ Features
- **Multi-Planet Support**: Simulate around Earth, Mars, Jupiter, Venus, Saturn
- **Real-time 3D CubeSat Visualization**: Projected cube with rotation, orbital trail
- **PID Controllers**: Independent roll/pitch/yaw stabilization with configurable gains
- **Orbital Mechanics**: Gravity, thrust correction, orbit decay/escape detection
- **Interactive GUI**:
  - Login screen (Demo: Username `Dinith`, Password `2002`)
  - Live attitude graphs (Roll/Pitch/Yaw)
  - Scrollable telemetry panels (position, velocity, attitude history)
  - Off-attitude warning log
  - Controls: Play/Pause/Stop/Reset, speed adjustment, keyboard disturbances
- **Auto-Stabilize**: One-click recovery from unstable attitudes/orbits
- **Responsive UI**: Resizable window, planet details popups

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- **Windows Users**: Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for Pygame

### Installation
```bash
pip install pygame numpy simple-pid
```

### Run
```bash
python system.py
```

## 🎮 Usage
1. **Login**: Use `Dinith` / `2002`
2. **Select Planet**: "Change Planet" → Pick Earth/Mars/etc.
3. **Start Simulation**: Click "Start/Resume"
4. **Controls**:
   | Action | Key/Button |
   |--------|------------|
   | Disturb Attitude | Arrow Keys |
   | Speed ± | Slow/Fast buttons |
   | Stabilize | "Auto-Stabilize" (when unstable) |
   | Toggle Panels | History/Sat/Warnings |
5. **Monitor**: Watch telemetry, graphs, warnings

## 🐛 Troubleshooting
- **Pygame Install Fail (Windows)**: Install C++ Build Tools → Restart terminal → Retry `pip install pygame`
- **Import Errors**: See [TODO.md](TODO.md) for status
- Verify deps: `python -c "import pygame, numpy, simple_pid; print('OK')"`

## 📈 Screenshots
*(Recommended: Capture login, main sim, graphs, planet menu)*

## 🔧 Development
- Main file: [system.py](system.py)
- Edit PID gains in `AttitudeController` / `OrbitalController`
- Add planets to `PLANET_MODELS`

## 📋 TODO
See [TODO.md](TODO.md) for dependency fixes and enhancements.

## 🤝 Contributing
Fork → PR with `blackboxai/` branch prefix.

## 📄 License
MIT License - see [LICENSE](LICENSE) *(create if needed)*

**Author**: Dinith (CubeSat Team)  
**Repo**: [DM-GITHUB/CubeSat](https://github.com/DM-GITHUB/CubeSat)

