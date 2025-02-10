from ctypes import CDLL, POINTER, c_int, c_float, c_void_p, byref
import os
import platform

def get_lib_path():
    # Determine system and architecture
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map architecture names
    arch_mapping = {
        'x86_64': 'x86_64',
        'amd64': 'x86_64',
        'arm64': 'aarch64',
        'aarch64': 'aarch64',
    }

    # Map system names
    system_mapping = {
        'darwin': 'macos',
        'linux': 'linux',
        'windows': 'windows'
    }

    # Get standardized names
    arch = arch_mapping.get(machine)
    sys_name = system_mapping.get(system)

    if not arch or not sys_name:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")

    # Determine library file extension
    if system == 'darwin':
        lib_ext = '.dylib'
    elif system == 'linux':
        lib_ext = '.so'
    elif system == 'windows':
        lib_ext = '.dll'
    else:
        raise RuntimeError(f"Unsupported system: {system}")

    # Construct library name
    lib_filename = f"libquadtree{lib_ext}"

    # Construct path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(
        script_dir,
        "zig-rtree",
        "zig-out",
        "lib",
        f"{arch}-{sys_name}",
        lib_filename
    )

    if not os.path.exists(lib_path):
        raise RuntimeError(f"Library not found at {lib_path}")

    return (lib_path, lib_filename)


(lib_path, lib_filename) = get_lib_path()

_lib = CDLL(lib_path)
# try:
#     _lib = CDLL(lib_path)
# except OSError:
#     # Fallback to system paths if not found in current directory
#     _lib = CDLL(lib_filename)

# Set up the argument and return types for each function
_lib.quadtree_init.argtypes = [c_float, c_float, c_float]
_lib.quadtree_init.restype = c_void_p

_lib.quadtree_deinit.argtypes = [c_void_p]
_lib.quadtree_deinit.restype = None

_lib.quadtree_reset.argtypes = [c_void_p, c_float, c_float, c_float]
_lib.quadtree_reset.restype = None

_lib.quadtree_set_threshold.argtypes = [c_void_p, c_float]
_lib.quadtree_set_threshold.restype = None

_lib.quadtree_set_force_params.argtypes = [c_void_p, c_float, c_float]
_lib.quadtree_set_force_params.restype = None

_lib.quadtree_insert.argtypes = [c_void_p, c_float, c_float, c_float]
_lib.quadtree_insert.restype = c_int

_lib.quadtree_remove.argtypes = [c_void_p, c_float, c_float, c_float]
_lib.quadtree_remove.restype = c_int

_lib.quadtree_get_total_mass.argtypes = [c_void_p]
_lib.quadtree_get_total_mass.restype = c_float

_lib.quadtree_get_force.argtypes = [
    c_void_p, c_float, c_float, c_float,
    POINTER(c_float), POINTER(c_float),
]
_lib.quadtree_get_force.restype = None


class QuadTree:
    """Python wrapper for the Zig Quadtree implementation."""

    def __init__(self, side_len: float = 0, center: tuple[float, float] = (0.0, 0.0)):
        """Initialize a new quadtree with given center coordinates and side length."""
        (x, y) = center
        self._ptr = _lib.quadtree_init(c_float(side_len), c_float(x), c_float(y))
        if not self._ptr:
            raise MemoryError("Failed to create quadtree")

    def __del__(self):
        """Clean up the quadtree when the Python object is destroyed."""
        if hasattr(self, '_ptr') and self._ptr:
            _lib.quadtree_deinit(self._ptr)
            self._ptr = None

    def reset(self, side_len: float, center: tuple[float, float] = (0.0, 0.0)):
        """Reset the quadtree with new parameters."""
        (x, y) = center
        _lib.quadtree_reset(self._ptr, c_float(side_len), c_float(x), c_float(y))

    def set_threshold(self, threshold: float):
        """Set the threshold for large body approximation."""
        _lib.quadtree_set_threshold(self._ptr, c_float(threshold))

    def set_force_params(self, c: float, r: float):
        """Set the force equation parameters."""
        _lib.quadtree_set_force_params(self._ptr, c_float(c), c_float(r))

    def insert(self, position: tuple[float, float], mass: float = 1.0) -> None:
        """Insert a point with mass into the quadtree."""
        (x, y) = position
        result = _lib.quadtree_insert(self._ptr, c_float(x), c_float(y), c_float(mass))
        if result != 0:
            raise Exception("failed to insert body into quadtree")

    def remove(self, position: tuple[float, float], mass: float = 1.0) -> None:
        """Remove a point with mass from the quadtree."""
        (x, y) = position
        result = _lib.quadtree_remove(self._ptr, c_float(x), c_float(y), c_float(mass))
        if result != 0:
            raise Exception("failed to remove body from quadtree")

    def get_total_mass(self) -> float:
        """Get the total mass stored in the quadtree."""
        return _lib.quadtree_get_total_mass(self._ptr)

    def get_force(self, position: tuple[float, float], mass: float = 1.0) -> tuple[float, float]:
        """Get the force exerted on a body."""
        (x, y) = position
        result_x = c_float()
        result_y = c_float()
        _lib.quadtree_get_force(self._ptr, x, y, mass, byref(result_x), byref(result_y))
        return (result_x.value, result_y.value)

# Example usage
if __name__ == "__main__":
    # Create a quadtree centered at (0,0) with side length 100
    tree = QuadTree(100.0)

    # Insert some points
    tree.insert((10.0, 10.0), 10000)
    tree.insert((-10.0, -10.0), 20000)

    # Get total mass
    print(f"Total mass: {tree.get_total_mass()}")

    print("force", tree.get_force((12, 10)))
    tree.set_force_params(-2, -3)
    print("force", tree.get_force((12, 10)))

    # Remove a point
    tree.remove((10.0, 10.0), 10000)

    # Get updated total mass
    print(f"Total mass after removal: {tree.get_total_mass()}")
    tree.reset(100.0)
    print(f"Total mass after reset: {tree.get_total_mass()}")
