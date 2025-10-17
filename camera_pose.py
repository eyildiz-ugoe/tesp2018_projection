from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pygame
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from pygame import mixer

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Settings:
    """Runtime configuration for the interactive projection."""

    projected_height: int = 1080
    projected_width: int = 1920
    min_match_count: int = 6
    max_match_count: int = 20
    matrix_smoothing_factor: float = 0.2
    delta_t: float = 1.0
    info_location: Optional[Tuple[float, float]] = None
    marker_points: Optional[Tuple[Tuple[int, int], ...]] = None
    camera_width: int = 1600
    camera_height: int = 896

    def __post_init__(self) -> None:
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        if self.info_location is None:
            self.info_location = (self.projected_width * 0.2, self.projected_height * 0.7)
        if self.marker_points is None:
            self.marker_points = (
                (0, 0),
                (0, self.projected_height - 100),
                (self.projected_width - 100, 0),
                (self.projected_width - 100, self.projected_height - 100),
            )

    def update_projection(self, height: int, width: int) -> None:
        self.projected_height = height
        self.projected_width = width
        self.info_location = None
        self.marker_points = None
        self._ensure_defaults()


@dataclass
class TrackingState:
    """Stores the current state of the shuttle tracking calculations."""

    center_point: np.ndarray
    velocity: np.ndarray
    homography: np.ndarray


@dataclass
class PlanetTemplateImage:
    """Holds a grayscale template image and its identifier."""

    name: str
    image: np.ndarray

    @classmethod
    def from_path(cls, path: Path) -> "PlanetTemplateImage":
        image = cv2.imread(str(path), 0)
        if image is None:
            raise FileNotFoundError(f"Could not load template image at {path!s}")
        return cls(name=path.stem, image=image)


@dataclass
class Planet:
    """Container describing a celestial body."""

    name: str
    distance_from_earth: str
    size: str
    number_of_moons: str
    gravity: str
    compounds_found: str
    orbit_time: str
    day_time: str
    surface_temperature: str

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> "Planet":
        values = [child.text or "" for child in element]
        return cls(*values)


def prepare_info(planet: Planet) -> str:
    """Format a block of descriptive text for the provided planet."""

    return (
        "----------Celestial Body Info"
        f"\n--Name: {planet.name}"
        f"\n--Distance from the Earth: {planet.distance_from_earth} kilometers"
        f"\n--Size: {planet.size} x of Earth"
        f"\n--Gravity: {planet.gravity} x of Earth"
        f"\n--Number of Moons: {planet.number_of_moons}"
        f"\n--Compounds Found: {planet.compounds_found}"
        f"\n--orbit Time: {planet.orbit_time} Earth days"
        f"\n--Day Time: {planet.day_time} Earth days"
        f"\n--Surface Temperature: {planet.surface_temperature} Degrees Celcius"
    )


def get_planet_pixel_locations(
    background_image: np.ndarray,
    templates: Sequence[PlanetTemplateImage],
) -> List[Tuple[str, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Return bounding boxes for each template found in the background image."""

    grayscale = cv2.cvtColor(background_image.copy(), cv2.COLOR_BGR2GRAY)
    planet_rects: List[Tuple[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = []

    for template in templates:
        width, height = template.image.shape[::-1]
        res = cv2.matchTemplate(grayscale, template.image, cv2.TM_SQDIFF_NORMED)
        _, _, min_loc, _ = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        planet_rects.append((template.name, (top_left, bottom_right)))

    return planet_rects


def init_webcam(
    primary_index: int,
    fallback_index: int,
    desired_width: int,
    desired_height: int,
) -> Tuple[cv2.VideoCapture, int, int]:
    """Attempt to connect to the webcam, falling back to a secondary index."""

    def _open_camera(index: int) -> Tuple[cv2.VideoCapture, np.ndarray]:
        camera = cv2.VideoCapture(index)
        camera.set(cv2.CAP_PROP_FPS, 50)
        camera.set(cv2.CAP_PROP_EXPOSURE, 10)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        ret_val, frame = camera.read()
        if not ret_val or frame is None:
            raise RuntimeError(f"Unable to read from camera index {index}.")
        return camera, frame

    errors = []
    tried_indices = []
    for index in (primary_index, fallback_index):
        if index in tried_indices:
            continue
        tried_indices.append(index)
        try:
            camera, frame = _open_camera(index)
            height, width, _ = frame.shape
            return camera, height, width
        except RuntimeError as exc:  # pragma: no cover - depends on hardware
            errors.append(str(exc))

    joined_errors = " \n".join(errors)
    raise RuntimeError(
        "No available camera detected. Attempted indices: "
        f"{', '.join(str(idx) for idx in tried_indices)}.\n{joined_errors}"
    )


def get_feature_matches(
    matcher: cv2.BFMatcher,
    detector: cv2.ORB,
    projection_descriptors: np.ndarray,
    camera_image: np.ndarray,
    max_matches: int,
) -> Tuple[List[cv2.DMatch], Sequence[cv2.KeyPoint]]:
    """Return feature matches between the projection and camera images."""

    keypoints = detector.detect(camera_image, None)
    if not keypoints:
        return [], []

    keypoints, descriptors = detector.compute(camera_image, keypoints)
    if descriptors is None or projection_descriptors is None:
        return [], keypoints

    matches = matcher.match(projection_descriptors, descriptors)
    if len(matches) > max_matches:
        matches = sorted(matches, key=lambda match: match.distance)[:max_matches]

    return matches, keypoints


def get_homography(
    matches: Sequence[cv2.DMatch],
    projection_keypoints: Sequence[cv2.KeyPoint],
    camera_keypoints: Sequence[cv2.KeyPoint],
    min_match_count: int,
) -> Tuple[np.ndarray | None, Sequence[int] | None]:
    """Calculate the homography matrix mapping projection points to camera points."""

    if len(matches) <= min_match_count:
        print(f"Not enough matches are found - {len(matches)}/{min_match_count}")
        return None, None

    src_pts = np.float32([projection_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([camera_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist() if mask is not None else None
    return homography_matrix, matches_mask


def show_matches(
    projection_image: np.ndarray,
    camera_image: np.ndarray,
    projection_keypoints: Sequence[cv2.KeyPoint],
    camera_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    matches_mask: Sequence[int] | None,
    homography_matrix: np.ndarray,
) -> None:
    """Visualise matched key points between the projection and camera images."""

    height, width, _ = projection_image.shape
    pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography_matrix)
    annotated_camera_image = cv2.polylines(camera_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=2,
    )

    visualization_image = cv2.drawMatches(
        projection_image,
        projection_keypoints,
        annotated_camera_image,
        camera_keypoints,
        matches,
        None,
        **draw_params,
    )
    cv2.imshow("Debug", visualization_image)


def virtual_point(
    homography_matrix: np.ndarray,
    camera_width: int,
    camera_height: int,
) -> np.ndarray:
    """Return the projector coordinates that correspond to the camera centre."""

    pts = np.float32([[round(camera_width / 2), round(camera_height / 2)]]).reshape(-1, 1, 2)
    _, inverse_matrix = cv2.invert(homography_matrix)
    return cv2.perspectiveTransform(pts, inverse_matrix)


def smoothen_matrix(
    current_matrix: np.ndarray,
    new_matrix: np.ndarray,
    smoothing_factor: float,
) -> np.ndarray:
    """Smooth large changes in the homography matrix to prevent jitter."""

    return current_matrix * smoothing_factor + new_matrix * (1 - smoothing_factor)


def is_image_fully_visible(
    homography_matrix: np.ndarray,
    background_height: int,
    background_width: int,
) -> bool:
    """Return True if the detected projection corners form a plausible rectangle."""

    pts = np.float32(
        [[0, 0], [0, background_height - 1], [background_width - 1, background_height - 1], [background_width - 1, 0]]
    ).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography_matrix)
    area = cv2.contourArea(dst)
    if area <= 100000:
        return False

    x, y, width, height = cv2.boundingRect(dst)
    metric = width * height
    error_ratio = abs(metric - area) / area
    return error_ratio < 0.8


def smoothen_center_motion(
    measured_position: Sequence[float],
    delta_t: float,
    state: TrackingState,
    smoothing_factor: float,
) -> None:
    """Update the tracked centre point using a simple motion model."""

    predicted = state.center_point + state.velocity * delta_t
    measured = np.array(measured_position, dtype=np.float32)
    smoothed = np.round(predicted * smoothing_factor + measured * (1 - smoothing_factor)).astype(np.int32)
    state.velocity = smoothed - state.center_point
    state.center_point = smoothed


def transparent_overlay(
    background_image: np.ndarray,
    overlay_image: np.ndarray,
    position: Tuple[int, int] = (0, 0),
    scale: float = 1.0,
) -> np.ndarray:
    """Overlay a (potentially) transparent image on top of the background image."""

    if scale != 1:
        overlay_image = cv2.resize(overlay_image, (0, 0), fx=scale, fy=scale)

    rows, cols, _ = background_image.shape
    height, width = overlay_image.shape[:2]
    y, x = position

    if x >= rows or y >= cols:
        return background_image

    overlay_slice = overlay_image[: max(0, rows - x), : max(0, cols - y)]
    height, width = overlay_slice.shape[:2]
    roi = background_image[x : x + height, y : y + width]

    if overlay_slice.shape[2] == 3:
        alpha_mask = np.ones((height, width, 1), dtype=np.float32)
        overlay_rgb = overlay_slice.astype(np.float32)
    else:
        alpha_mask = (overlay_slice[:, :, 3:] / 255.0).astype(np.float32)
        overlay_rgb = overlay_slice[:, :, :3].astype(np.float32)

    background_rgb = roi.astype(np.float32)
    blended = alpha_mask * overlay_rgb + (1 - alpha_mask) * background_rgb
    roi[:] = blended.astype(np.uint8)
    return background_image


def get_camera_rotation(
    homography_matrix: np.ndarray,
    camera_width: int,
    camera_height: int,
) -> float:
    """Return the estimated rotation angle of the camera relative to the projection."""

    camera_pts = np.float32(
        [
            [round(camera_width / 2), round(camera_height / 2)],
            [round(10 + camera_width / 2), round(10 + camera_height / 2)],
        ]
    ).reshape(-1, 1, 2)
    proj_pts = cv2.perspectiveTransform(camera_pts, homography_matrix)

    camera_vector = np.array(
        [
            camera_pts[0][0][0] - camera_pts[1][0][0],
            camera_pts[0][0][1] - camera_pts[1][0][1],
        ],
        dtype=np.float32,
    )
    proj_vector = np.array(
        [
            proj_pts[0][0][0] - proj_pts[1][0][0],
            proj_pts[0][0][1] - proj_pts[1][0][1],
        ],
        dtype=np.float32,
    )

    camera_vector /= max(np.linalg.norm(camera_vector), 1e-6)
    proj_vector /= max(np.linalg.norm(proj_vector), 1e-6)

    sin_angle = camera_vector[0] * proj_vector[1] - camera_vector[1] * proj_vector[0]
    angle = np.arcsin(np.clip(sin_angle, -1.0, 1.0))
    return float(angle)


def is_inside_rect(
    current_pos: Sequence[int],
    top_left: Sequence[int],
    bottom_right: Sequence[int],
) -> bool:
    """Determine whether the provided position lies within the rectangle."""

    return (
        top_left[0] <= current_pos[0] <= bottom_right[0]
        and top_left[1] <= current_pos[1] <= bottom_right[1]
    )


def clean_asin(asin_angle_in_radians: float) -> float:
    """Clamp the input to the valid range of arcsin to avoid NaNs."""

    return float(min(1, max(asin_angle_in_radians, -1)))


def load_planet_data(xml_path: Path) -> List[Planet]:
    """Load celestial body metadata from the provided XML file."""

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    planets: List[Planet] = []
    for bodies in root:
        entries = bodies.findall("planet") or bodies.findall("star")
        for entry in entries:
            planets.append(Planet.from_xml_element(entry))

    if not planets:
        raise ValueError(f"No celestial bodies were loaded from {xml_path!s}.")

    return planets


def load_planet_templates(template_dir: Path) -> List[PlanetTemplateImage]:
    """Load and index the planet templates used for collision detection."""

    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory {template_dir!s} does not exist.")

    templates: List[PlanetTemplateImage] = []
    for path in sorted(template_dir.glob("*.png")):
        templates.append(PlanetTemplateImage.from_path(path))

    if not templates:
        raise ValueError(f"No templates were found in {template_dir!s}.")

    return templates


def start_background_music(sound_path: Path, *, loop: bool = True) -> bool:
    """Initialise the audio mixer (if possible) and start the background music."""

    try:
        if not pygame.mixer.get_init():  # pragma: no branch - trivial guard
            pygame.mixer.init()
        mixer.music.load(str(sound_path))
        mixer.music.play(-1 if loop else 0)
        return True
    except pygame.error as exc:  # pragma: no cover - depends on audio hardware
        print(f"Audio disabled: {exc}")
        return False


def create_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Return a truetype font, falling back to the default if unavailable."""

    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        return ImageFont.load_default()


def apply_markers(
    projection_image: np.ndarray,
    marker_paths: Sequence[Path],
    marker_points: Sequence[Tuple[int, int]],
) -> None:
    """Overlay the tracking markers on top of the projection image."""

    for marker_path, point in zip(marker_paths, marker_points):
        marker_image = cv2.imread(str(marker_path))
        if marker_image is None:
            raise FileNotFoundError(f"Marker image {marker_path!s} could not be loaded.")
        height, width, _ = marker_image.shape
        x, y = point
        projection_image[y : y + height, x : x + width] = marker_image.copy()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Interactive Solar System projection controller.")
    parser.add_argument("--camera-index", type=int, default=1, help="Primary camera index to use.")
    parser.add_argument("--fallback-camera-index", type=int, default=0, help="Fallback camera index if the primary fails.")
    parser.add_argument("--mute", action="store_true", help="Disable background music.")
    parser.add_argument(
        "--resource-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory containing templates, sounds and other resources.",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable the debug visualisation window.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings()

    resource_root = args.resource_root.resolve()
    templates_dir = resource_root / "templates"
    planet_templates = load_planet_templates(templates_dir)

    if not args.mute:
        background_track = resource_root / "sounds" / "background.mp3"
        if background_track.exists():
            start_background_music(background_track)
        else:
            print("Background track not found; continuing without audio.")

    planets = load_planet_data(resource_root / "planet_info.xml")

    try:
        camera, camera_height, camera_width = init_webcam(
            args.camera_index,
            args.fallback_camera_index,
            desired_width=settings.camera_width,
            desired_height=settings.camera_height,
        )
    except RuntimeError as exc:
        print(exc)
        return 1

    projection_path = resource_root / "solar_system2.png"
    projection_image = cv2.imread(str(projection_path))
    if projection_image is None:
        raise FileNotFoundError(f"Projection image {projection_path!s} could not be loaded.")

    shuttle_path = resource_root / "shuttleIcon.png"
    shuttle_icon = cv2.imread(str(shuttle_path), cv2.IMREAD_UNCHANGED)
    if shuttle_icon is None:
        raise FileNotFoundError(f"Shuttle icon {shuttle_path!s} could not be loaded.")

    marker_files = [
        resource_root / "markers" / "marker_one_small.png",
        resource_root / "markers" / "marker_two_small.png",
        resource_root / "markers" / "marker_three_small.png",
        resource_root / "markers" / "marker_four_small.png",
    ]

    projection_height, projection_width = projection_image.shape[:2]
    settings.update_projection(projection_height, projection_width)
    apply_markers(projection_image, marker_files, settings.marker_points)

    projection_detector = cv2.ORB_create(nfeatures=500)
    projection_keypoints = projection_detector.detect(projection_image, None)
    projection_keypoints, projection_descriptors = projection_detector.compute(
        projection_image, projection_keypoints
    )

    if not projection_keypoints or projection_descriptors is None:
        raise RuntimeError("Failed to compute keypoints for the projection image.")

    planet_locations = get_planet_pixel_locations(projection_image, planet_templates)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    camera_detector = cv2.ORB_create(nfeatures=500)

    font = create_font(resource_root / "spacefont.ttf", 20)

    tracking_state = TrackingState(
        center_point=np.array([settings.projected_width // 2, settings.projected_height // 2], dtype=np.int32),
        velocity=np.zeros(2, dtype=np.int32),
        homography=np.identity(3, dtype=np.float32),
    )

    try:
        cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Projector", projection_image)
        if not args.no_debug:
            cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

        while True:
            processed_image = projection_image.copy()
            ret_val, camera_image = camera.read()
            if not ret_val or camera_image is None:
                print("Failed to read from camera. Exiting.")
                break

            matches, camera_keypoints = get_feature_matches(
                matcher,
                camera_detector,
                projection_descriptors,
                camera_image,
                settings.max_match_count,
            )

            homography_matrix, matches_mask = get_homography(
                matches,
                projection_keypoints,
                camera_keypoints,
                settings.min_match_count,
            )

            if homography_matrix is None:
                cv2.imshow("Projector", processed_image)
                cv2.waitKey(10)
                continue

            if not args.no_debug:
                show_matches(
                    processed_image,
                    camera_image,
                    projection_keypoints,
                    camera_keypoints,
                    matches,
                    matches_mask,
                    homography_matrix,
                )

            if is_image_fully_visible(homography_matrix, settings.projected_height, settings.projected_width):
                tracking_state.homography = smoothen_matrix(
                    tracking_state.homography,
                    homography_matrix,
                    settings.matrix_smoothing_factor,
                )
                virtual_point_location = virtual_point(
                    tracking_state.homography,
                    camera_width,
                    camera_height,
                )
                updated_point = virtual_point_location[0][0]
            else:
                updated_point = tracking_state.center_point

            smoothen_center_motion(
                updated_point,
                settings.delta_t,
                tracking_state,
                settings.matrix_smoothing_factor,
            )

            for template_name, (top_left, bottom_right) in planet_locations:
                if not is_inside_rect(tracking_state.center_point, top_left, bottom_right):
                    continue

                matching_planet = next(
                    (
                        planet
                        for planet in planets
                        if planet.name.lower() == template_name.lower()
                    ),
                    None,
                )
                if matching_planet is None:
                    continue

                info_text = prepare_info(matching_planet)
                img_pil = Image.fromarray(processed_image)
                draw = ImageDraw.Draw(img_pil)
                draw.text(tuple(map(int, settings.info_location)), info_text, font=font, fill=(0, 255, 255, 0))
                processed_image = np.array(img_pil)
                break

            rotated_shuttle = shuttle_icon.copy()
            rows, cols = rotated_shuttle.shape[:2]
            angle = get_camera_rotation(
                tracking_state.homography,
                camera_width,
                camera_height,
            )
            angle_in_degrees = math.degrees(clean_asin(angle))
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle_in_degrees, 1)
            rotated_shuttle = cv2.warpAffine(rotated_shuttle, rotation_matrix, (cols, rows), cv2.INTER_LANCZOS4)

            processed_image = transparent_overlay(
                processed_image,
                rotated_shuttle,
                position=tuple(tracking_state.center_point.tolist()),
                scale=0.7,
            )

            cv2.imshow("Projector", processed_image)
            key = cv2.waitKey(20)
            if key in {ord("a"), ord("q")}:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
