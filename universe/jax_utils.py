
import jax
import jax.numpy as jnp

# Enable 64-bit precision (Essential for orbital mechanics)
jax.config.update("jax_enable_x64", True)

# Constants (km^3/s^2)
GM_SUN = 1.32712440018e11
GM_JUPITER = 1.26686534e8
GM_SATURN = 3.7931187e7
GM_IO = 5959.91
GM_EUROPA = 3202.73
GM_GANYMEDE = 9887.83
GM_CALLISTO = 7179.28

def get_gm_values():
    return jnp.array([
        GM_SUN,
        GM_JUPITER,
        GM_SATURN,
        GM_IO,
        GM_EUROPA,
        GM_GANYMEDE,
        GM_CALLISTO
    ])
