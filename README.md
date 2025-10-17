# Interactive Solar System

This is the repository for the TESP 2018 group project named "Interactive Solar System". The idea of the project is to provide an easy learning tool for the kids who could simply project a 2D image of the solar system on a surface, on which they could control a space shuttle with an external USB camera to roam around the space and land on the celestial bodies to get information about them.

## Requirements

The application now targets Python **3.10**. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

A USB camera should be connected in order to run the program. Audio playback is optional; if no audio device is present the program continues without background music.

## Usage

Run the program and follow the on-screen prompts:

```bash
python camera_pose.py
```

Useful command line options:

- `--camera-index` / `--fallback-camera-index` – override which camera device to open.
- `--mute` – skip loading and playing the background music.
- `--no-debug` – disable the debug window that displays feature matches.
- `--resource-root` – point the application at an alternative assets directory.

Direct the external USB camera towards the surface on which the projector is projecting the 2D image. The USB camera will act as a controller for the space shuttle, which should be moved gently and slowly. Abrupt movements and sudden changes in rotation may not be the most clever way to control the shuttle. As the shuttle lands on the celestial body, a small information box about the planets will pop up.

## Contributors

This project has been supervised by Professor Kagami at Tohoku University. The following members have equally contributed to the project progress:

- @Spimp
- @Enzymator
- @DesuDeluxe
- @eyildiz-ugoe

## Continuous Development

Depending on availability of free time, the project can be extended by one or more members of the project. The file `camera_pose2.py` is maintained as a compatibility wrapper around the primary entry point for anyone following older instructions.
