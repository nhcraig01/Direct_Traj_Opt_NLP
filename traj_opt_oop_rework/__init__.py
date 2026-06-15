import jax

# Required for correctness, not just precision: at production integration
# tolerances (a_tol = r_tol = 1e-12), diffrax's adaptive step controller
# cannot converge in float32 and the integrator pegs `max_steps`. Must be set
# before any other JAX arrays are created, so it lives at package import time
# rather than in each entry-point script.
jax.config.update("jax_enable_x64", True)
