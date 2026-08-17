Build a single self-contained HTML file called `index.html` containing a three.js
N-body orbital mechanics simulation. No build step, no bundler, no local
dependencies — load three.js from a CDN with an ES module import map.

## Physics requirements

- Simulate gravitational N-body interaction with real pairwise forces, not
  hard-coded circular paths. Every body must attract every other body.
- Integrate with **velocity Verlet**, not naive Euler. State the timestep in a
  comment and keep the integration stable.
- Seed the system with a central star and 5 planets on near-circular orbits.
  Derive each planet's initial tangential velocity from `v = sqrt(G * M / r)`
  so the orbits start stable rather than immediately decaying.
- Add one body on a deliberately eccentric orbit so the difference is visible.
- Use a softening term in the force calculation to avoid a singularity when two
  bodies pass very close.
- Display the total system energy (kinetic + potential) in a corner readout. It
  should stay roughly constant — that is the check that the integrator is correct.

## Visual requirements

- Perspective camera with orbit controls (drag to rotate, scroll to zoom).
- Each body rendered as an emissive sphere, sized by mass, with a distinct colour.
- Every body leaves a fading trail showing its recent path. Trails must be
  updated efficiently, not rebuilt from scratch every frame.
- A point light at the star, and enough ambient light that planets are not black.
- Starfield background.

## Interaction requirements

- Slider controlling simulation speed, from paused through to 10× real time.
- Button that resets the system to its initial state.
- Clicking a body focuses the camera on it and keeps it centred as it moves.
- Readout showing elapsed simulation time and current total energy.

## Constraints

- One file. Everything inline: HTML, CSS, JavaScript.
- Must run correctly when opened directly in a browser.
- No external assets — no texture files, no model files, no images.
- Handle window resize correctly.
- Do not use deprecated three.js APIs; target a current version and say which
  version you targeted in a comment at the top.

Output the complete file. Do not abbreviate any section or leave placeholder
comments like "rest of the code here".
