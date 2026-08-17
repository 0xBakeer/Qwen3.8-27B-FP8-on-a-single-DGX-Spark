Build a single self-contained HTML file called `index.html` containing a three.js
GPU particle flow-field simulation. No build step, no bundler, no local
dependencies — load three.js from a CDN with an ES module import map.

## Simulation requirements

- **200,000 particles minimum**, rendered as a single `THREE.Points` object with
  a custom `ShaderMaterial`. One draw call.
- Particle motion is driven by a **curl-noise flow field computed in GLSL**, not
  on the CPU. Write the noise function yourself in the shader — simplex or value
  noise is fine, but it must be in the shader source, not imported.
- Curl noise means: sample a potential field and take its curl, so the resulting
  velocity field is divergence-free and the particles swirl instead of clumping
  into sinks. Compute the curl by finite differences of the noise.
- Particles must have a finite lifetime. When a particle dies it respawns at a
  new position. Handle lifetime and respawn on the GPU using the vertex shader
  and a per-particle seed attribute, driven by a single `uTime` uniform.
- Particle colour varies with velocity magnitude, using a palette defined in the
  fragment shader. Faster particles should read as hotter.
- Additive blending, depth write disabled, so dense regions bloom naturally.

## Visual requirements

- Particles rendered as soft round points, not hard squares. Discard fragments
  outside the point radius and fade toward the edge.
- Dark background with a subtle vignette or gradient — not flat black.
- The camera slowly orbits the field on its own so the three-dimensional
  structure is visible without the user touching anything.
- Maintain interactive framerate. Say in a comment what you did to keep it fast.

## Interaction requirements

- Mouse movement distorts the flow field near the cursor position, pushing
  particles away. Pass the cursor in as a uniform.
- Keyboard or on-screen controls for: noise scale, flow speed, particle
  lifetime.
- An FPS counter, computed yourself rather than from a library.

## Constraints

- One file. Everything inline: HTML, CSS, JavaScript, GLSL as template strings.
- Must run correctly when opened directly in a browser.
- No external assets — no texture files, no model files, no images. The particle
  sprite must be generated procedurally in the fragment shader.
- Handle window resize correctly, including device pixel ratio.
- Do not use deprecated three.js APIs; target a current version and say which
  version you targeted in a comment at the top.

Output the complete file. Do not abbreviate any section or leave placeholder
comments like "rest of the code here".
