import math

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("pygame")

import numpy as np

import camera_pose


def test_prepare_info_contains_planet_name():
    planet = camera_pose.Planet(
        name="Earth",
        distance_from_earth="0",
        size="1",
        number_of_moons="1",
        gravity="1",
        compounds_found="Water",
        orbit_time="365",
        day_time="1",
        surface_temperature="15",
    )

    info = camera_pose.prepare_info(planet)
    assert "Earth" in info
    assert "Water" in info
    assert "million kilometers" in info


def test_clean_asin_clamps_values():
    assert camera_pose.clean_asin(2.0) == 1.0
    assert camera_pose.clean_asin(-2.0) == -1.0
    assert math.isclose(camera_pose.clean_asin(0.5), 0.5)


def test_smoothen_matrix_interpolates():
    current = np.identity(3, dtype=np.float32)
    new = np.ones((3, 3), dtype=np.float32)
    result = camera_pose.smoothen_matrix(current, new, 0.5)
    assert np.allclose(result, 0.5 * current + 0.5 * new)


def test_smoothen_center_motion_updates_state():
    state = camera_pose.TrackingState(
        center_point=np.zeros(2, dtype=np.int32),
        velocity=np.zeros(2, dtype=np.int32),
        homography=np.identity(3, dtype=np.float32),
    )
    camera_pose.smoothen_center_motion((10, 10), 1.0, state, 0.5)
    assert np.array_equal(state.center_point, np.array([5, 5]))
    assert np.array_equal(state.velocity, np.array([5, 5]))


def test_transparent_overlay_blends_images():
    background = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :, :3] = 255
    overlay[:, :, 3] = 128

    result = camera_pose.transparent_overlay(background.copy(), overlay, position=(1, 1))
    assert np.all(result[1:3, 1:3] == 128)


def test_is_inside_rect():
    assert camera_pose.is_inside_rect((5, 5), (0, 0), (10, 10))
    assert not camera_pose.is_inside_rect((11, 5), (0, 0), (10, 10))


def test_is_image_fully_visible_with_identity_matrix():
    assert camera_pose.is_image_fully_visible(
        np.identity(3, dtype=np.float32),
        camera_pose.Settings().projected_height,
        camera_pose.Settings().projected_width,
    )


def test_get_planet_pixel_locations_finds_template():
    background = np.zeros((20, 20, 3), dtype=np.uint8)
    background[5:9, 5:9] = 255
    template = camera_pose.PlanetTemplateImage(name="test", image=np.full((4, 4), 255, dtype=np.uint8))

    locations = camera_pose.get_planet_pixel_locations(background, [template])
    assert locations == [("test", ((5, 5), (9, 9)))]


def test_load_planet_data_and_templates():
    templates = camera_pose.load_planet_templates(camera_pose.PROJECT_ROOT / "templates")
    assert templates, "Expected at least one template to be loaded"

    planets = camera_pose.load_planet_data(camera_pose.PROJECT_ROOT / "planet_info.xml")
    assert planets, "Expected planet data to be loaded"
    assert any(planet.name.lower() == "earth" for planet in planets)
