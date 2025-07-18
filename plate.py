from collections.abc import Iterable
import numpy as np


# Helper functions
def unit_vector(v: np.ndarray) -> np.ndarray:
    # Normalizes a vector, unless already 0D-like
    # NOTE: np's divide-where, norm, and explicit float(64) cast are slow
    # An if-else + once-only norm + zeros_like almost halves decode runtime
    mag_v = np.linalg.norm(v)
    return v / mag_v if mag_v > 0 else np.zeros_like(v)


def find_mirror_point(
    a: np.ndarray,
    b: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    # This inverse of find_mirror()
    # NOTE: Not currently in use, and not sure when it will be
    # For two vectors and a mirror normal, find the natural reflection spot
    # We assume the vectors already subtract the plane center (so the plane now passes through 0,0,0)

    # First, flip b to the other side via projection to normal
    b_direction = plane_normal * np.dot(plane_normal, b) / np.dot(plane_normal, plane_normal)
    flipped_b = b - (2 * b_direction)

    # With "reflection" reframed as an "intersection", solve plane intersection
    # To do this, sub the parametric form of a line into the general form of a plane
    # We can simplify knowing the input is 0 centered, then solve for the magical "t"
    line_direction = flipped_b - a
    t_side = sum(plane_normal * line_direction)  # Gather up t/constant terms
    value_side = -sum(plane_normal * a)  # Element-wise mul, btw
    if t_side == 0:
        raise ValueError("Zero or multiple solutions, since ab is parallel to the plane.")
    t = value_side / t_side
    # I'll never remember how this works...

    # Plug scalar t back into the line's parametric form to get xyz
    return line_direction * t + a


def find_mirror(
    a: np.ndarray,
    b: np.ndarray,
    mirror_point: np.ndarray,
) -> np.ndarray:
    # For some source/dest point and a reflection point, find the mirror's normal
    # Will return all 0s if a/b too close to the mirror or on opposite sides

    # Shift source/dest points to be mirror-relative
    rel_a = a - mirror_point
    rel_b = b - mirror_point

    # Mirror's direction is the sum of relative (normalized) directions
    return unit_vector(unit_vector(rel_a) + unit_vector(rel_b))


def bulk_angle_between(
    unit_query: np.ndarray,  # n_dims
    unit_data: np.ndarray,  # n_vectors x n_dims (n_dims also works)
) -> np.ndarray:
    # Return the angle in radians between 2 nD vectors
    # Because arctan2 doesn't work for nD
    return np.arccos(np.clip(np.dot(unit_data, unit_query), -1.0, 1.0))


# Core classes
class Cell:
    def __init__(self, coords: np.ndarray):
        # Share fields by ref, acting as a simple struct
        self.__coords = coords  # NOTE: May consider removing in the future if unused, since coords are ints anyways
        self.__gradients = np.empty((0, 3))
        # May store hundreds of gradients per cell even for simple images

    @property
    def coords(self) -> np.ndarray:
        return self.__coords
    
    @property
    def gradients(self) -> np.ndarray:  # n_gradients x n_dims
        return self.__gradients
    
    @gradients.setter
    def gradients(self, gradients: np.ndarray) -> None:
        self.__gradients = gradients
    

class Plate:
    # Represents a collection of cells in 3D that can encode directional binary images
    # This version stores a simple rectangular grid of cells on z=0, facing 0,0,-1
    # Bounds/shift can be provided, but resolution is in integers

    # TODO: May consider hexagonal packed cells or cylindrical plates later
    # TODO: Does not handle self-occluding cell checks yet

    def __init__(
        self,
        start_x: int,
        start_y: int,
        size_x: int,
        size_y: int,
    ):
        if size_x <= 0 or size_y <= 0:
            raise ValueError("XY bounds must be a postive area.")
        
        self.__start_x = start_x
        self.__start_y = start_y
        self.__size_x = size_x
        self.__size_y = size_y

        # In Python, numpy arrays are not ideal for unvectorized iteration, so a
        # flattened, row-major list of Cell objs is marginally more efficient.
        # Almost all cells are touched in full-parallax, so store as dense.
        self.__cells = [
            Cell(np.array([x + start_x, y + start_y, 0]))
            for y in range(size_y)
            for x in range(size_x)
        ]

    def closest_cell(self, point: np.ndarray) -> Cell:
        # Returns the Cell the point correspond to, clamping if out-of-range
        # This interface might be useful for arbitrary orientation or shapes?
        adjusted_x = int(round(point[0])) - self.__start_x
        adjusted_y = int(round(point[1])) - self.__start_y
        adjusted_x = min(max(adjusted_x, 0), self.__size_x - 1)
        adjusted_y = min(max(adjusted_y, 0), self.__size_y - 1)

        return self.__cells[adjusted_y * self.__size_y + adjusted_x]
    
    def sightline_cell(
        self,
        camera: np.ndarray,
        keypoint: np.ndarray
    ) -> Cell | None:
        # Finds the cell that a keypoint's sightline corresponds to
        # Returns None if zero or multiple solutions

        # Since the plate is defined flat at z=0, we can massively simplify
        if camera[2] == keypoint[2]:
            return None  # Parallel, hence no solutions
        else:
            # Solve for unknown t in line vector form
            line_direction = camera - keypoint
            t = -keypoint[2] / line_direction[2]  # Just solve with parametric z

            # Plug scalar t back into the line's vector form to get xyz
            raw_coords = line_direction * t + keypoint

            # Match this arbitrary plane point to a nearby cell
            # Assumes integer cell pos and negligible internal cell structure
            cell = self.closest_cell(raw_coords)
            if np.linalg.norm(cell.coords - raw_coords) <= 1.415:  # Bit more than a diagonal
                return cell
            else:
                return None  # Out of grid bounds

    def encode_plate(
        self,
        encoding_data: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        # Engraves new gradients into the plate's cells
        # Input is some list or iter yielding:
        # - current point light source (3D vector)
        # - current point camera (3D vector)
        # - frame at this perspective (zero or more 3D vectors)

        # Queue changes to build arrays only once
        new_gradients: dict[Cell, list[np.ndarray]] = {}

        # For now, we know the plate exists on 0,0,0 with normal 0,0,-1 with integer cell pos
        # We also know that source/camera pos must be -z
        for source, camera, frame in encoding_data:
            for keypoint in frame:
                # Locate the plate's cell the keypoint corresponds to
                cell = self.sightline_cell(camera, keypoint)

                if cell is not None:  # Only if it lands on the plate
                    # Compute and store the true mirror angle required
                    if cell in new_gradients:
                        new_gradients[cell].append(find_mirror(source, camera, cell.coords))
                    else:
                        new_gradients[cell] = [find_mirror(source, camera, cell.coords)]

        # Resolve queued changes
        for cell, gradients in new_gradients.items():
            cell.gradients = np.vstack(gradients)

    def decode_plate(
        self,
        source: np.ndarray,
        camera: np.ndarray,
        rad_tol: float = 0.017453,  # Roughly 1 degree
    ) -> list[np.ndarray]:
        # Returns all plate coords that should light up from a perspective
        # (the render will figure out the global to screen space placement via camera properties)
        # Subclass extensions may want to account for self-occlusion by the plate or cell engraving shape
        visible_points = []

        for cell in self.__cells:
            # Check if this cell has suitable gradient to create a reflection
            # If it's close enough, append coords to output
            expected = find_mirror(source, camera, cell.coords)
            if len(cell.gradients) > 0 and np.any(bulk_angle_between(
                expected,
                cell.gradients,
            ) <= rad_tol):
                visible_points.append(cell.coords)

        return visible_points
