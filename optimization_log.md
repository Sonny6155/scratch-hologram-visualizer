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