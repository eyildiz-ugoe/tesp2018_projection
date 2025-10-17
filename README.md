# 🚀 Interactive Solar System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An interactive educational tool for learning about the solar system through augmented reality projection. Control a virtual space shuttle using a USB camera to explore planets and learn about celestial bodies!

## 📖 About

This is the repository for the **TESP 2018 group project** at Tohoku University. The project provides an engaging learning experience for children by:

- 🎯 Projecting a 2D solar system onto any surface
- 📹 Using a USB camera as a motion controller
- 🛸 Navigating a virtual space shuttle through space
- 🪐 Landing on planets to discover fascinating information
- 🎵 Enjoying an immersive audio experience

## ✨ Features

- **Camera-based Control**: Your USB camera becomes a space shuttle controller
- **Interactive Learning**: Hover over planets to reveal educational content
- **Real-time Tracking**: Advanced computer vision using OpenCV ORB feature detection
- **Smooth Movement**: Interpolated tracking for fluid shuttle motion
- **Educational Content**: Detailed information about all solar system planets
- **Customizable**: Multiple command-line options for different setups

## 🎉 Recent Updates (2025)

The project has been **completely modernized** for Python 3.10+:

- ✅ Modern Python syntax with type hints
- ✅ Comprehensive test suite (9 unit tests)
- ✅ Improved error handling and stability
- ✅ Command-line interface with flexible options
- ✅ Better code organization and maintainability
- ✅ Full backward compatibility

## 📋 Requirements

### Hardware
- 🖥️ Computer with Python 3.10 or later
- 📹 USB camera (external or built-in)
- 📽️ Projector
- 🔊 Audio output (optional)

### Software Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy >= 1.21` - Numerical computations
- `opencv-python >= 4.5` - Computer vision and feature detection
- `Pillow >= 8.0` - Image processing
- `pygame >= 2.0` - Audio playback and multimedia

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/eyildiz-ugoe/tesp2018_projection.git
cd tesp2018_projection

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Basic usage
python camera_pose.py

# With options
python camera_pose.py --mute --no-debug

# Get help
python camera_pose.py --help
```

## 🎮 How to Use

1. **Setup**: Position your projector to display the solar system image on a flat surface
2. **Camera**: Point your USB camera at the projected surface
3. **Control**: Move the camera gently to control the space shuttle
4. **Explore**: Navigate to different planets and celestial bodies
5. **Learn**: When the shuttle lands on a planet, information appears automatically!

> 💡 **Tip**: Move the camera slowly and smoothly for best results. Abrupt movements may cause tracking issues.

## ⚙️ Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--camera-index` | Primary camera device index | `1` |
| `--fallback-camera-index` | Backup camera if primary fails | `0` |
| `--mute` | Disable background music | `False` |
| `--no-debug` | Hide debug visualization window | `False` |
| `--resource-root` | Custom directory for assets | Current directory |

### Examples

```bash
# Use built-in camera (index 0)
python camera_pose.py --camera-index 0

# Silent mode without debug window
python camera_pose.py --mute --no-debug

# Custom resource location
python camera_pose.py --resource-root /path/to/assets
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=camera_pose

# Run specific test
pytest tests/test_camera_pose.py::test_prepare_info_contains_planet_name
```

**Test Results**: All 9 tests passing ✅

## 🏗️ Project Structure

```
tesp2018_projection/
├── camera_pose.py          # Main application (modernized)
├── camera_pose2.py         # Legacy compatibility wrapper
├── planet_info.xml         # Celestial body data
├── requirements.txt        # Python dependencies
├── solar_system2.png       # Projection background image
├── shuttleIcon.png         # Space shuttle sprite
├── spacefont.ttf          # Custom font for text display
├── markers/               # ArUco-style tracking markers
├── sounds/                # Background music and audio
├── templates/             # Planet template images for detection
└── tests/                 # Unit test suite
    └── test_camera_pose.py
```

## 🔧 How It Works

1. **Feature Detection**: Uses OpenCV ORB (Oriented FAST and Rotated BRIEF) to detect keypoints in the projected image
2. **Homography Calculation**: Computes the transformation matrix between camera view and projected image
3. **Position Tracking**: Maps camera center point to virtual position on the projected surface
4. **Motion Smoothing**: Applies velocity-based interpolation for fluid movement
5. **Collision Detection**: Template matching to detect when shuttle enters planet boundaries
6. **Information Display**: Renders educational content from XML data when landing occurs

## 🪐 Celestial Bodies Included

The application includes detailed information about:

- ☀️ **Sun** - Our star
- ☿️ **Mercury** - The swift planet
- ♀️ **Venus** - The morning star
- 🌍 **Earth** - Our home
- ♂️ **Mars** - The red planet
- ♃ **Jupiter** - The giant
- ♄ **Saturn** - The ringed planet
- ♅ **Uranus** - The ice giant
- ♆ **Neptune** - The blue planet

Each celestial body displays:
- Distance from Earth
- Relative size and gravity
- Number of moons
- Atmospheric composition
- Orbit and day duration
- Surface temperature

## 🐛 Troubleshooting

### Camera Not Found
```bash
# Try different camera indices
python camera_pose.py --camera-index 0
python camera_pose.py --camera-index 1
```

### No Audio Output
```bash
# Run in mute mode
python camera_pose.py --mute
```

### Poor Tracking Performance
- Ensure good lighting conditions
- Keep the projected image fully visible to the camera
- Move the camera slowly and steadily
- Check that markers are clearly visible

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

## 👥 Contributors

This project was supervised by **Professor Kagami** at Tohoku University during the TESP 2018 summer school program.

**Team Members** (equal contribution):
- [@Spimp](https://github.com/Spimp)
- [@Enzymator](https://github.com/Enzymator)
- [@DesuDeluxe](https://github.com/DesuDeluxe)
- [@eyildiz-ugoe](https://github.com/eyildiz-ugoe)

## 🔄 Continuous Development

The project is maintained and can be extended by team members. The modernized codebase makes it easier to add new features:

- The file `camera_pose2.py` serves as a compatibility wrapper for legacy documentation
- New features should be added to `camera_pose.py`
- Tests should be added to `tests/test_camera_pose.py`
- All changes should maintain backward compatibility

## 📝 License

This project is open source. Please check the LICENSE file for more information.

## 🙏 Acknowledgments

- **Tohoku University** for hosting the TESP 2018 summer school
- **Professor Kagami** for supervision and guidance
- **OpenCV community** for the excellent computer vision library
- **Python community** for the amazing ecosystem

## 📚 Learn More

- [TESTING_REPORT.md](TESTING_REPORT.md) - Comprehensive testing documentation
- [PR_STATUS.md](PR_STATUS.md) - Modernization PR details
- [planet_info.xml](planet_info.xml) - Celestial body data structure

## 🚀 Future Ideas

Potential enhancements for future development:
- [ ] Add more celestial bodies (dwarf planets, asteroids)
- [ ] Implement 3D visualization
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] VR/AR integration
- [ ] Online multiplayer mode
- [ ] Achievement system for students

---

**Made by the TESP 2018 team**

*Last updated: October 2025*
