

CONCEPT:
I focused on both geometry and topology.
Geometry: The coral structure is defined using rectangular segments that form branches growing upwards from a base. The branch length and number of segments per branch control the overall shape of the coral.
Topology: Connections are created between parent and child segments using springs, ensuring that branches remain attached and sway together instead of separating. The branching behavior introduces hierarchical connectivity, mimicking a tree-like structure.
The combination of rectangular particles + connections allows the coral to be both visually accurate and dynamically interactive in the MPM simulation.

IMPLEMENTATION:
Core Function: build_coral(scene, ...)
This function:
Creates a base (scene.add_rect(...)) as a foundation.
Adds an initial root segment to act as the first branching node.
Recursively generates branches using grow_branch(...).
Attaches branches to their parent nodes via scene.add_connection(parent_id, new_id).
Recursive Function: grow_branch(...)
Creates a limited number of segments per branch.
Recursively grows new branches at different angles.
Ensures each segment is connected to its parent to maintain structure.
PERFORMANCE
Speed:
The original implementation crashed due to excessive memory use (too many particles).
The optimized version reduces segment count & depth, which greatly improves runtime performance.
Stability:
Originally, branches weren’t connected properly and collapsed separately.
By explicitly connecting segments via scene.add_connection(), the structure remains intact and moves in a realistic, swaying motion.
 Memory Usage:
Reduced branch depth and limited the number of segments to prevent CUDA out-of-memory (OOM) errors.
Capped actuator count to prevent over-allocation of GPU memory.
