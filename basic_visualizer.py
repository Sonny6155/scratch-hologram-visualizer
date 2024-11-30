from collections.abc import Iterable
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plate import Plate, unit_vector


def spiral_scenario() -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # Build a discrete sampling of frame data and corresponding viewing angles
    # Rotate 1 to 179 degrees across 178//4 = 44 frames
    for frame_i in range(0, 179, 4):
        # Using a "following light" setup, rotate around y-axis at r=50
        # The plate visually spins anti-clockwise relative, per bird's eye
        # Also give it a slight overhead swing to break ortho ambiguity
        radian_angle = (frame_i + 1) * np.pi / 180
        source = np.array([50*np.cos(radian_angle), 10*np.sin(radian_angle), -50*np.sin(radian_angle)])
        camera = source.copy()

        # Animated spiral grows from 10 deep at r=0, to plate surface at r=10
        # This spans 3 rotations over depth 10
        spiral = []
        t = (frame_i + 1) / 45  # 4 animated rotations over the plate rotation span
        for depth in range(91):
            depth_t = t - depth/30
            depth_r = 10 - depth/9
            spiral.append(np.array([depth_r*np.cos(2*np.pi*depth_t), depth_r*np.sin(2*np.pi*depth_t), depth]))

        yield source, camera, np.array(spiral)


def sine_scenario() -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # Rotate 1 to 179 degrees across 178//4 = 44 frames
    for frame_i in range(0, 179, 4):
        # Using a "following light" setup, rotate around y-axis at r=50
        radian_angle = (frame_i + 1) * np.pi / 180
        source = np.array([50*np.cos(radian_angle), 10*np.sin(radian_angle), -50*np.sin(radian_angle)])
        camera = source.copy()

        # Mark surface corners to show normal depth
        sine_kps = [[-15, -15, 0], [15, 15, 0], [-15, 15, 0], [15, -15, 0]]

        # Animated sine wave, deep within the plate
        t = (frame_i + 1) / 4
        for i in range(-20, 21):
            sine_kps.append(np.array([i*2, 10*np.sin(t - i/4), 100]))
        # Unfortunately, the animation actually makes it hard to perceive
        # depth, despite the super deep inset

        yield source, camera, np.array(sine_kps)


# TODO: A complex Morlet wavelet projection with equal protrusion and depth will be so cool, and absolutely MUST be done


if __name__ == "__main__":
    plate = Plate(-20, -20, 41, 41)  # Spans -20 to 20 in x/y, facing 0,0,-1

    # TODO: need to import object data later for fixed object testing
    # TODO: also want to import a binary thresholded, decimated gif animation, then render centered on zero

    # Build our aligned perspective and frame data
    # Frames hold keypoints in world-space, but only from certain perspectives
    # TODO: This info remains uncompressed, but am considering a generator for encoding
    # perspectives, frames = sine_scenario()
    # print("scene set up done")

    # Encode to the plate
    start = time.time()
    # Feed in generator to sort of half main scope memory
    plate.encode_plate(spiral_scenario())
    end = time.time()
    print(f"encoding done, took {end - start:.8f}s")

    # NOTE: Used to test unencoded angles or the effect of moving just the light etc. Comment out when unneeded
    # perspectives_list = []
    # for y in range(0, 11, 1):
    #     for frame_i in range(0, 179, 4):
    #         # Using a "following light" setup, rotate around y-axis at r=50
    #         # The plate visually spins anti-clockwise relative, per bird's eye
    #         # Also give it a slight overhead swing to break ortho ambiguity
    #         radian_angle = (frame_i + 1) * np.pi / 180
    #         source = np.array([50*np.cos(radian_angle-0.1), y, -50*np.sin(radian_angle-0.1)])
    #         camera = np.array([50*np.cos(radian_angle), y, -50*np.sin(radian_angle)])
    #         perspectives_list.append((source, camera, None))

    # Results show that keeping the "following light" assumption is safe, but
    # fixing the light (like to [0,10,-50]) for follow-encoded data causes
    # flickering, destroy the depth. Slight differences in following angle is
    # mostly safe, to a point.

    # Simulate visualization (as matplotlib animation)
    # Points are directly projected to unnormalized viewport 
    fig, ax = plt.subplots()
    ims = []
    for source, camera, _ in spiral_scenario():
        # Compute new screen space basis, oriented up (+/-z if flat)
        # Yanked from my quantum geo x-means code a while back...
        camera_normal = unit_vector(-camera)  # Looking at 0,0,0
        if camera_normal[0] == 0 and camera_normal[2] == 0:
            # If plane is vertically flat, default new y to the z-axis
            screen_x = np.array([1, 0, 0])
            screen_y = np.array([0, 0, -camera_normal[1]])
        else:
            # Else, the new y is the steepest +ve direction on the plane
            # Double cross product does the trick: (norm X up) X norm
            # Reordered to retain x's handed-ness on (+xy when viewing at +z)
            screen_x = np.cross([0, 1, 0], camera_normal)
            screen_y = np.cross(camera_normal, screen_x)
            # Since y is always +ve, x direction is consistent

            screen_x = unit_vector(screen_x)
            screen_y = unit_vector(screen_y)

        # Draw the plate's hardcoded bounding box to projected screen space
        all_bb_x = [
            np.dot([-20, 20, 0], screen_x),
            np.dot([20, 20, 0], screen_x),
            np.dot([20, -20, 0], screen_x),
            np.dot([-20, -20, 0], screen_x),
            np.dot([-20, 20, 0], screen_x),
        ]
        all_bb_y = [
            np.dot([-20, 20, 0], screen_y),
            np.dot([20, 20, 0], screen_y),
            np.dot([20, -20, 0], screen_y),
            np.dot([-20, -20, 0], screen_y),
            np.dot([-20, 20, 0], screen_y),
        ]
        plot_data = ax.plot(all_bb_x, all_bb_y, animated=True)  # List of Line2D according to docs

        # Decode a single perspective and scatterplot to screen space
        all_x = []
        all_y = []
        start = time.time()
        decoded_points = plate.decode_plate(source, camera)
        for point in decoded_points:
            all_x.append(np.dot(point, screen_x))
            all_y.append(np.dot(point, screen_y))
        plot_data.extend(ax.plot(all_x, all_y, "bo", animated=True))

        ims.append(plot_data)
        end = time.time()
        print(f"decoded a perspective, took {end - start:.8f}s")

    # Animate
    for line in ims[0]:  # Initial frame must be explicitly drawn, not blitted
        line.set_animated(False)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()
