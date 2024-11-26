# scratch-holography-visualizer
The aim of this project is to simulate and visualize scratch/specular
holograms for arbitary plane/light/camera settings.

This will primarily focus on researching a rasterized implementation, but may
extend the work to generate and visualize both continuous and rasterized arcs.

## Background
Conventional holography transforms a controlled wavefront of light into an
arbitrary new wavefront, potentially down to the phase. The traditional method
can effectively store the lightfield almost perfectly (given enough money).

Meanwhile, parallax-based methods filter what is visible based on perspectives
through a barrier or a angle-selective magnifying lens, and work well for
light-emitting technologies or in ambient light without perfectly recreating
the input wavefront. A distinction is usually made between conventional
holography and parallax methods, though both are valid forms of multiscopy.

Specular holography is a unique class of multiscopy different to (but shares
many characteristics of) parallax-based methods and conventional holography.
Unlike the traditional methods, it operates through geometric optics from
well-positioned "mirrors", and arises naturally in polished surfaces as glints
or potnetially entire multiscopic rings. Mechanically, it is similar or maybe
even identical to embossed methods like rainbow holograms. It also happens to
have like 20 names for literally the exact same thing, "scratch holography"
being merely the most common one...

## Points of Exploration
Two types will be investigated:
- The original arc type
    - Circular arcs are all that is required to generate good horizontal rotation (and frankly that gives the best parallax illusion anyways)
    - Does not support full-parallax (natively)
        - The double reflection problem can be resolved with an angled scribe, but not easily for generic view angles
    - Technically preserves horizontal phase while scrambling vertical, but we will ignore this feature
    - Artefacting may occur with densely stacked arcs
        - Though the level of artifacting greatly depends on the scribe precision and number of stacks (scene density)
    - Wide arcs can generate small line spans instead of precisely localized glints
        - This can be partially mitigated by controlling the light source's spatial coherence, and producing more "geometrically ideal" scratch walls
- Microwells (singular, controlled scratch pixels)
    - Generalizes arcs to localized points of a dense gradient set, akin to hogels
    - Has the potential to implement full-parallax
        - Certain configurations can trade multi-observer robustness for a full, 180 degree viewing angle
    - Can be arranged as regular pixels or compacted even further in a honeycomb arrangement
    - May reduce artifacting at arc ends and dense intersections, and improve glint localization
        - However, the rasterized nature requires a higher "resolution" for emulating some smooth movements
    - Unresolved manufacturing feasibility
        - Blocking out an angle requires somehow painting the corresponding reflection surface
        - Alternatively, cutting 90 degrees down or abrasing the surface could work to absorb/scatter light in an embossable manner
        - As a challenge, may also consider the possibility of RGB reflections

## Current Progress
The generalized encoding visualizer is sort of done, but the visual part needs to be moved to an interactive, perspective 3D viewer with fine-grained config in tkinter.
This shows that full-parallax is indeed "probably possible", given the light source and camera are identically positioned
(or at least positioned relatively under certain assumptions).
Did some testing on changing source and plate angle for a different perspective encoding, and it broke as expected.
There is a tiny bit of flexibility offered at sufficient distances, say 0.1 radians difference between light source and camera or slight position shift,
but it does break down eventually with only a ~1 degree of gradient viewing tolerance per cell.
Will stick to following light for now to maximize viewing angle for, though at
the cost of even the limited angular multiscopy offered by Beaty's or Duke's approaches.

Further work:
- Will need a way to import arbitrary vertex or wireframe data for visualizing static 3D models through standard formats (STL? OBJ?).
- May also need to further optimize depending on performance when scaling keypoints.
- Would like to automate arc generation for static images using Duke's method for fixed source, spinning 45 degree viewer.
    - The midpoint circle algorithm is a good starting point, though volatile keypoints requires modification to only rasterize arcs.
- May consider a way of associating fixed points and drawing its catmull-rom movement spline as a purely rendering feature.
- Will also consider derived Plate classes for dense hexagonal packing, and maybe cylindrical?

## Visualization Implementation (WIP)
The default plate such that model space mostly equals world space.
A desired, discrete frame is encoded into the plate at each viewing perspective,
aka some (preferrably distant) light source and camera position.
The engraved images can then be replayed by "decoding" at some perspective.
This can be the same source/camera angles as during encoding, or completely
arbitrary ones for experimentation.

GUI behaviour should:
- start user at x/y = 0 at radius 50 facing towards 0,0,0 (so pos at 0,0,-10)
- allow user to rotate the camera while facing 0,0,0 around the current radius
- allow user to adjust radius
- do not currently allow user to rotate arbitrarily? (lock their relative up vector to always be y >= 0)
- allow user to reset to face 0,0,0 or reset both pos and direction
- allow user to click to draw a point? or maybe have this be imported obj or point file?

## Physical Implementation
One thought is to use a series of "microwells", as inspired by microlens but using reflection techniques rather than magnification.
Compartmentalizing cells this way should avoid the minor artifacting in increasingly dense scenes under traditional scratch holography,
where stacking arcs would damage the encoded gradients over the intersection.
Additionally, it could help localize glints better or at least shape them to spec, whereas large arcs would often create a short glint line.
These theoretical benefits still require some further testing.

Rather, the more useful property is as an alternate precision manufacturing option that could minimise travel distance for arbitrary scenes, if it can be manufactured reliably.
We won't be able to encode phase very easily, but that doesn't matter for simple multiscopy under incoherent lighting.

Having the light source and camera come from roughly the same point location should increase viewing angle of an ideal well to 180 degree,
whereas wide source-camera angles would be affected by well geometry, akin to self-occlusion in microlens.
For a simple wall display, a 45 degree fixed light source and moving camera should be sufficient for wide horizontal viewing and "mostly-full-parallax".

The slight distortion of inexact source and camera in the real-world can be mitigated by standing back a bit as to approach orthographic-like FoV.

## Limitations
The current implementation reads discrete frames and stores the required gradient (mirror normal) per frame.
This unfortunately means that it must read and store info about every keypoint for every viewing angle setup,
even if the data can be parametrically described as a simple, continuous motion and stored as an exact curve.

Additionally, viewing (decoding) the picture assumes a small angle tolerance.
From a physical perspective, this depends on the manufacturing technique, precision of well encoding, well size etc.
