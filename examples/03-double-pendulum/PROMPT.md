Build a single self-contained HTML file called `index.html` containing a three.js
double-pendulum chaos simulation. No build step, no bundler, no local
dependencies — load three.js from a CDN with an ES module import map.

## Physics requirements

- Implement the **exact double-pendulum equations of motion**, derived from the
  Lagrangian. Do not use a small-angle approximation and do not fake it with
  two independent sine waves.
- Integrate with **RK4**. Fixed timestep, with an accumulator so the physics rate
  is independent of the render framerate.
- Simulate **8 pendulums simultaneously**, identical except that each starting
  angle is perturbed by an additional 1e-6 radians. This is the point of the
  piece: they track each other exactly, then visibly diverge.
- Show the divergence quantitatively — a readout of the angular spread between
  the first and last pendulum, and the elapsed time at which the spread first
  exceeds 0.1 radians.
- Masses and rod lengths must be adjustable, and changing them must rebuild the
  simulation cleanly rather than corrupting the running state.

## Visual requirements

- Each pendulum drawn with rods and two bobs, in a colour ramp across the eight
  so divergence is easy to follow.
- The tip of each pendulum traces a fading path. Older path segments fade out.
  Update trails efficiently rather than rebuilding geometry every frame.
- Slight perspective camera with orbit controls, and a subtle ground grid for
  spatial reference.
- Soft shadows if they can be made cheap enough; otherwise say in a comment why
  you skipped them.

## Interaction requirements

- Drag either bob of the first pendulum to reposition it; releasing restarts all
  eight from that configuration with their perturbations reapplied.
- Sliders for rod lengths, masses, gravity, and trail length.
- Pause, step-one-frame, and reset controls.
- A small 2D phase-space plot, drawn on a `<canvas>` overlay, showing θ₁ against
  its angular velocity for the first pendulum.

## Constraints

- One file. Everything inline: HTML, CSS, JavaScript.
- Must run correctly when opened directly in a browser.
- No external assets — no texture files, no model files, no images.
- Handle window resize correctly, including the overlay canvas.
- Do not use deprecated three.js APIs; target a current version and say which
  version you targeted in a comment at the top.

Output the complete file. Do not abbreviate any section or leave placeholder
comments like "rest of the code here".
