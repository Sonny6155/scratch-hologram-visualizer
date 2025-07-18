Unfortunately, decoding isn't the fastest because it tries to be realistic by checking every cell.
Not all of it can be vectorized, plus it would potentially degrade maintainability.
We need decoding, specifically, to be fast enough on CPU for tkinter,
where we will later explore pushing some work to the GPU to migrate from informal ortho to a more accurate perspective view.
Or just straight up raytracing.

Presumably, tkinter only requires like 100ms ticks, but we still want to be as fast as possible.
Plus, we will be storing not just a linear sweep of perspectives, but potentially a whole landscape by the end

Some tricks may be helpful later, and indicate new issues,
so will try to keep some misc notes of discovered optimizations and results.

## Original on 41x41 plate
Avg frame decode time:
- 1 point: ~50ms
- 11 points: ~59ms
- 31 points: ~76ms
- 91 points: ~100ms

Observations:
- There is a tiny performance increase (like milliseconds) when frames show a lot of nodes, maybe because of short-circuited gradient search.
    - Short of some smart sorting or even explicit randomization, it's unlikely short-circuiting can be leverged further.
- These is also potential for gradient storage asymptotic improvement using kd or octrees, but the base overhead and difficulty is likely just worse.
    - Tying into this will be the need to encode for all full-parallax perspectives and such.
    - One lazy optimisation is to convert list to array, at the cost of degraded multiple encode steps performance (which we assume we don't even need).
    - Another one is to run a prune function to avg out overly similar after all encoding is done, since duplicates occassionally occur (maybe model-based clustering weighted to size?).
- The biggest saving is if the constant time overhead of raw cell iteration can be improved from a whopping 50ms. Gradient search is only semi-bad of this sparse data in comparison.

## Unit Vector Optimization
Avg frame decode time:
- 1 point: ~32ms
- 11 points: ~37ms
- 31 points: ~47ms
- 91 points: ~66ms

Observations:
- This change reduced pretty much everything equally, because everything needs to normalise vectors at some point.
- Timing singular np functions can be unrealiable, presumably due to caching or batching? Sometimes it's 0s, then spotaneously spikes 1ms.
    - Solution is to measure on a larger scale. In this case, only observe the runtime of a single plate decode.
- Simple assignment costs nothing, so always consider caching. Building new np arrays or running their functions may take much longer.
- The focus should now be on gradient search, probably employing cached h/v angles to reduce several divisions in angle comparisons.
    - Should be like 2 subtracts, an add, a multiply, and a less-than on a constant squared distance. Ignoring all the python-side overhead, I guess.
    - Gradient space will roughly asymptotically double, but that's a small price for already small data.
    - This will also allow us to semi-vectorise angle similarity, though to what effect is unknown. Note how short-circuiting was found to be ineffective on arbitrary ordering.

## Assessing Existing Naive Implementation
### Problem
Revisiting the current implementation with a minimal example,
the current encoding can just barely handle a very simple full-parallax use case.
However, some more pratical testing will be required to assess the runtime as it scales,
plus the space usage is a bit worrying.

Consider the following parameters for a bare minimum product:
- 50 horizontal * 50 vertical viewpoints = 2500 perspectives
    - This is a tad more than my early 44x1 experiments, but may vary depending on real-life/simulated usage
- 40 x 40 = 1600 plate cells (already visibly low resolution)
- Simple, static 3D object with just over 100 keypoints -> ~100 gradients per perspective
    - Each gradient is a normal vector of normalized xyz as a tuple

This results in a plate of 2500 x 100 x 3 = = 750,000 floats distributed around the 1600 cells.
Each cell should therefore average 2500 x 100 / 1600 = 156.25 unit vectors.
Given that floats in Python are by default doubles, this is at minimum 48MB
but likely much more when including overhead of collections and Python.

Time-wise, the current implementation linearly searches with short-circuiting, then computes delta angle every single time.
However, short-circuit halved avg runtime only applies when the item is definitely in the list and is randomized.
In our case, we know what a majority of the image will miss (roughly 1600 - 100 missed cells of varying sparseness).
So its best to assume that short-circuiting is useless and that it needs to search nearly all 250,000 vectors.
This has also been shown in early single row tests, where short-circuiting without randomization would only save some ~2 of 70ms.

There is the additional limitation that decoding should be doable within a 100ms render window,
much of which is likely to be CPU-dependent (and we are still waiting for stable free-threading for multi-core).
A couple trials show that 100ms is actually extremely easy to hit with multiple vertical data.

### Optimization factors
A big factor will be removing duplicates and close points. Removing exact duplicates with floats is easy.
However, tiny differences cannot be trivially removed, and must be handled heuristically by definition.
Perhaps this would be something like solving the closest pair of points problem on loop until a min distance,
or by single-linkage clustering followed by mean point aggregation.
Alternatively, storing in a tree with a distance limit or as scaled ints can
help enforce equal binning and be potentially much faster.

There are existing inefficiencies with how we compare angles each cell and each iteration.
For one, we are not using vectorization due to storing them in a regular list.
Also, in our current use case, we have not needed to actually use the normals,
rather only checking that a similar angle within a degree exists.
We could instead reduce xyz unit vectors into just h/v angles.
The lookup will either implement wraparound in the 360 case, or more likely
ignore it since surely no real-world plate has that much curvature.

There isn't much we can do about iterating every cell due to the point source/camera, and arbitrary source-camera angles support.
This also means that trees will not perform well when gradients are distributed across cells,
resulting in an increasingly O(kp_total + cells), rather than O(log(kp_total) + cells).
However, we do know that gradients often concentrate strongly at hotspots.
While it is unlikely that most data structures can take advantage of this variable sparseness,
it does allow trees to function not quite as terribly as expected in many cases.

Scaling grid resolution means that the existing keypoints get distributed among more cells.
This adds the overhead of empty cells, but at least doesn't add anymore actual keypoint compute time.
Therefore, runtime for most grouped-by-cells approaches should be O(kp + cells).

### Solutions considered
- Linear vectorization with precomputed angles
    - Abandon short-circuiting and dynamic append in favour of fixed datatypes and fast eager.
    - Precomputing the independent h/v components and doing only squared
    diagonal comparisons means that vectorization is trivial while skipping a
    ton of divisions.
    - Replacing normals with angles can save space. We can re-derive normals later with minor loss.
    - It is still a linear lookup.
        - This is unlikely to be a problem until we hit many hundreds of gradients per cells.
        Regardless, it still needs practical testing.
- Octree on xyz normals
    - Does not address the 3 floats space issue, rather adds more space due to
    requiring 8 children for every node.
    - Does not come with common packages.
- Kd-tree on h/v angles
    - Sklearn and scikit both have a C-space version.
        - We don't need to implement any wraparound in our limited use case,
        given we start from 0 degrees.
    - Kd-trees can get unbalanced if adding iteratively, which degrades lookup time.
    - No wasted child nodes like quadtrees/octrees.
    - Support nearest neightbour lookup by default in avg O(log n) time.
- Int h/v angles in hashset lookup
    - Ensures equal angle spacing by the nature of fixed resolution angles.
    - Naturally handles duplicates by rounding.
    - Neighbour lookup just by searching known nearby indices.
        - However, int degrees inherently creates 0 to 1 whole degree extra
        tolerance between the rounded query and stored vectors, on top of the
        intended tolerance.
        - This diagonal neighbours also add more degrees than cardinals at a
        low int resolution.
        - At higher resolution int scales, it must index a circular area.
    - Using int degrees is intuitive, but demanding a higher resolution must
    break this (like storing 1000 ints between 180 etc).
        - Needs to be rebuilt for adjusting viewing scale live.
    - Int sizes will likely be bloated at runtime, since we can't use
    fixed-size, non-object ints like with numpy arrays.
        - However, the tradeoff of fewer max and avg keypoints will be worth
        it when upscaling.
        - Export can still be made more efficient in a few ways.
    - Given the resolution loss and all the little technical details required,
    it's best not to implement this until cells reach some 1000 vectors.

## Angle Between Optimization
Removed redundant unit vector conversion in angle comparison for a hefty boost.
Tried replacing numpy clip with a Python-space scalar comparison, which gave a slight boost.
But added it back for the eager implementation, as it scales much better in bulk (only few milliseconds lost total),
while keeping the error-less guarantee.
Eventually settled on vectorized eager.

Avg frame decode time:
- 1 point: ~30ms
- 11 points: ~34ms
- 31 points: ~36ms
- 91 points: ~39ms

Additional tests:
- Full-parallax spiral (91 kp/frame, 44^2 = 1936 angles, 40^2 = 1600 cells)
    - Encoding took 4.1353s
    - Decoding took ~85ms per frame
- Full-parallax spiral (91 kp/frame, 44^2 = 1936 angles, 60^2 = 3600 cells)
    - Encoding took 5.0542s
    - Decoding took ~165ms per frame

Observations:
- The main overhead was computing redundant unit vectors. It also prevented
(easy) vectorization. Fixing this alone led to a massive speedup.
    - However, the base 30ms of calculating gradients of all cells seems much
    messier painful to remove.
- Eager was initially slower, but seemed to scale better as the number of
keypoints grew such that a few cells contained 100+ gradients.
This does seem to "prove" the ineffectiveness of short-circuiting here,
but more testing will be required as we scale up massively.
- In full-parallax, the gradient distribution turned out to be much more even.
    - In the tested spiral, numbers ranged somewhat evenly between 0 and 175
    gradients per cell, rather than mostly 0 with a handful of 40s and 100s.
    - This might not hold if the object is overly small or animated/encoded
    with a heavy directional skew.
    - Time-wise, it still doesn't scale that well increasing resolution.
    But 40^2 is good enough for rendering, and encoding is still impressive.
    - This increased distribution and poor resolution scaling suggests that
    tree lookup won't be that effective, so will skip them for now.

## Changing Collection Dtypes
Changing the plate's internal dtype from a 2D np array of Cell objs to a
flattened native list for Python-space iterations only gave a very slight
improvement, but did get rid of the nditer jank at least.

Storing Cell gradients as a 2D array during encoding (used for a vectorized
query in decode) instead of redudantly coercing a list of 3-vectors per Cell,
every decode, gives a massive decode improvement (~0.08s -> ~0.05s). This is
hopefully fast enough to comfortably render full-parallax for small plates
in interactive 3D later.

Observations:
- Might be worth moving to a saner language before attempting 3D, since:
    - Encoding still creates many small 3D vectors just for simple np calcs.
    - Decoding still has list iterations and various manual np call overheads.