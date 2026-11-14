# Spec: data movies (truth only, v1)

Status: ready for implementation. Owner of the spec: planner. Implementer: follow it exactly. Anything that
is not specified here should be raised as a question, not decided by the implementer.

## Purpose

1. Let the group *see* each dataset across its whole time range, with every train / validation / test
   chunk and excluded gap marked. That lets us check that the interleaved chunks sample the same
   physical states as training, and shows any drift over the run.
2. Build the frame renderer once, so the evaluation pipeline can later reuse it for truth / prediction /
   residual movies. Same geometry, same colour scales, same layout.

## Inputs (read-only; never modify the data)

- Split files: `splits/rc300_reduced_v1.json`, `splits/5000run_v1.json` (schema_version 2). Take every
  path from the split file and never list directories yourself.
  - The path of a dump is `data_root` + its entry in `splits.<split>.files`, where `<split>` is one of
    `train`, `validation`, `test`, `excluded`. Build one dump-ID → path map from all four lists.
  - The geometry path is `data_root` + `geometry_file`.
  - The movies and the timeline cover the **whole run**, excluded dumps included. Iterate over
    `segments`, which lists every dump exactly once in chronological order. Each segment has `chunk`
    (e.g. `train_3`, `validation_2`, `test_1`, `excluded_4`), `split`, `first`, `last` and `count`.
  - The v1 layout is 13 non-excluded chunks, alternating train and held-out:
    `train_1, validation_1, train_2, test_1, train_3, test_2, train_4, validation_2, train_5,
    validation_3, train_6, test_3, train_7`, with a 10-dump excluded segment on each side of every
    validation and test chunk.
- `geometry.npz` keys used: `r`, `h` (theta, in radians), `ph` (phi, in radians), each `[Nr, Nθ, Nφ]`;
  `spin`; `dt_dump`; `gdet`, `dx` (for the mass integral); and `resolved` (rc300 only). No quantity in
  this spec may be computed on a shell with r < 6.11 for rc300, apart from drawing the slices themselves.
- Dump arrays: `[8, Nr, Nθ, Nφ]` float32, channels `rho, ug, utilde1, utilde2, utilde3, B1, B2, B3`,
  physical (unnormalized) values. Load them with `np.load(path, mmap_mode="r")`.
- Simulation time of dump k: `t = t0 + k * dt_dump`.
  - For rc300, read `t` from `splits.json` in the data root (per-dump times, e.g. dump 3000 at
    t = 300007.4). Fall back to `k * dt_dump` only if a time is missing.
  - For 5000run, use `t = k * 25`.

## Coordinates (HARM-style reconstruction)

Plot on the physical (r, θ, φ) grid using the cell-centre coordinates from `geometry.npz`. Never plot
against index space.

- **Meridional slice (x–z plane):**
  - Take φ index `j0 = argmin |φ − 0|` and `j1 = (j0 + Nφ/2) mod Nφ`, which is φ ≈ π.
  - Right half: `x = r sinθ`, `z = r cosθ`, values at `j0`.
  - Left half: `x = −r sinθ`, `z = r cosθ`, values at `j1`.
  - Draw both halves with `pcolormesh(x, z, v, shading="gouraud")`. A seam on the axis is expected,
    since rc300 crops θ to 1.875°–178.125°.
- **Equatorial slice (x–y plane):**
  - Both datasets have an even Nθ (rc300: 48; 5000run: 96), so take the mean of the two θ rows nearest
    π/2, at indices `Nθ/2 − 1` and `Nθ/2`.
  - Plot with `x = r cosφ`, `y = r sinφ`.
  - Append the φ = 0 column again at φ = 2π so the disk closes with no wedge.
- Draw the event horizon as a filled black disk of radius `r_H = 1 + sqrt(1 − a²)`, with a = `spin`.
- **rc300 only:** outline the coarse AMR ring (`resolved == False`, r < 6.11) with a thin dashed white
  circle at its outer radius, labelled "coarse AMR level (r < 6.11)".
  - In that ring H-AMR ran at refinement level 0, half the resolution of the rest of the grid.
  - Preprocessing placed it on the finest grid by duplicating each level-0 value into 2 × 2 × 2 output
    cells.
  - The values are real simulation output, but at half resolution.
  - Source: `Fall26/docs/rc300_preprocessing.md`, section 3 and the `fill_coarse` setting.
- Use equal aspect ratio and axes in units of R_g.

## Quantities

| Name | Plotted value | Colormap |
|---|---|---|
| `rho` (required) | log10(rho) | `inferno` |
| `ug` (optional, `--quantity ug`) | log10(ug) | `inferno` |
| `beta_inv` (later, not v1) | needs b² from the metric; out of scope | — |

## Fixed colour limits (no per-frame autoscaling)

- Compute one pair (vmin, vmax) per (dataset, quantity, zoom), shared by both views so that one
  colourbar is valid for both panels. Pool the plotted values of **both** the meridional and the
  equatorial slice cells inside the zoom window, from **every 50th dump across the whole run**
  (dumps `first, first + 50, …` of the whole run, excluded dumps included). Use the 0.5th and 99.5th percentiles of the
  pooled values (`np.percentile`, default linear method), as float64.
- JSON layout: `{"<quantity>": {"<zoom>": {"vmin": …, "vmax": …, "dumps_sampled": N}}}`.
- Save them to `movies/<split_id>/colour_limits.json` and reuse them. If that file exists, read it
  rather than recompute.
- They are only for visualization, so using all splits is fine here. Never reuse these numbers as
  model statistics.

## Zooms

| Zoom name | Extent | Why |
|---|---|---|
| `inner` | \|x\|, \|z\| ≤ 50 R_g | inner disk and turbulence (the long-term science target) |
| `full` | \|x\|, \|z\| ≤ 1000 R_g (rc300 crop edge ≈ 1048) | the whole domain, outflow, large-scale drift |

## Frame layout (identical for every frame and dataset)

- Canvas 1920 × 1080 px: `figsize=(19.2, 10.8)`, `dpi=100`, matplotlib `Agg` backend, white background,
  font size 14.
- Left panel: meridional slice. Right panel: equatorial slice. One shared horizontal colourbar labelled
  e.g. `log10 ρ`.
- Title line: `<dataset> · dump <k> · t = <t:.0f> R_g/c · <CHUNK>`. `<CHUNK>` is the chunk name in
  capitals with a space, e.g. `TRAIN 3`, `VALIDATION 2`, `TEST 1`, `EXCLUDED 4`. Colour it by split:
  train `#2b6cb0`, validation `#c05621`, test `#c53030`, excluded `#718096`.
- Bottom timeline strip, full width:
  - a horizontal bar covering the whole dump range, with every segment shaded in its split colour;
  - a vertical marker at the current dump;
  - a tick at the first dump of each validation and test chunk, labelled with the short name (`V1`,
    `V2`, `V3`, `T1`, `T2`, `T3`) and its dump ID.
- Nothing else on the frame: no timestamps of the render, no usernames, no paths.

## Frame stride and encoding

| Dataset | Stride (dumps per frame) | Frames | Δt per frame | Length at 30 fps |
|---|---|---|---|---|
| rc300_reduced | 4 | 2,419 | 400 R_g/c | ≈ 81 s |
| 5000run | 2 | 2,857 | 50 R_g/c | ≈ 95 s |

- **Frame rule (this is normative; the counts above follow from it):** the frame set is the sorted union
  of two sets:
  - the on-stride dumps `first, first + stride, …`, up to and including the last dump of the run;
  - the first dump of every non-excluded segment (all 13 train, validation and test chunks).

  Excluded segments get no extra frame, and the last dump of the run is not added if it is off-stride.
  - rc300: 2,409 on-stride frames + 10 off-stride chunk starts = 2,419 frames.
  - 5000run: 2,851 on-stride frames + 6 off-stride chunk starts = 2,857 frames.
- Assert the computed frame count equals the table's count for these two split files.
- Render PNG frames into a temporary directory, then encode:
  `ffmpeg -y -framerate 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow <out>.mp4`
- On Delta and DeltaAI, run `module load ffmpeg/6.1.1` first.
- Delete the PNGs after a successful encode. Keep 7 still frames: the first frame of the run and the
  first frame of each of the 6 validation and test chunks.

## Timeline figure (required, also answers "is the run stationary?")

One PNG per dataset, `timeline.png`, with every dump (no stride) on the x axis (t in R_g/c) and the
every segment shaded in its split colour (the same colours as the frames, including excluded).
Label the validation and test chunks `V1`–`V3` and `T1`–`T3` along the top axis. Four stacked panels,
computed in physical units:

1. Rest mass in the band 6.11 ≤ r < 100: Σ rho · gdet · dx1 · dx2 · dx3 over cells with
   6.11 ≤ r < 100. Use the same band for **both** datasets so the column means the same thing.
   - For rc300, also assert that every cell in the band has `resolved == True`. Fail if not; don't
     silently mask.
   - Column name: `mass_r6_100`.
2. Mass flux (Ṁ): not in v1, because `utilde1` is not u^r and a metric-correct Ṁ needs the full
   4-velocity. Leave the panel out and add a one-line note under the figure: "Ṁ panel planned for v2".
   (So the figure has three panels in v1.)
3. Disk-averaged log10(rho) for 10 < r < 100, |θ − π/2| < 0.3.
4. Mean |B1| on the shell `i8 = argmin |r[:, 0, 0] − 8|` (a magnetic-flux proxy). Use r ≈ 8 for both
   datasets: it lies outside rc300's copied region (r < 6.11) and is still in the inner disk. Average
   |B1| over all θ and φ cells of that shell with equal weight (no gdet weighting in v1). Name the
   column `B1abs_r8`.

`timeline.csv` columns in v1, in this order: `dump,t,split,chunk,mass_r6_100,logrho_disk,B1abs_r8` (`split` is train / validation / test /
excluded; `chunk` is the segment's `chunk` name).

Also save the raw series as `timeline.csv` (columns `dump,t,split,<quantity>…`), so nobody needs to
reread 149 GB to replot.

## Outputs (exact names)

```
movies/<split_id>/                            e.g. movies/rc300_reduced_v1/
  <dataset>_<quantity>_<zoom>_s<stride>.mp4   e.g. rc300_reduced_rho_inner_s4.mp4
  stills/<dataset>_<quantity>_<zoom>_<dump>.png
  colour_limits.json
  timeline.png
  timeline.csv
  movie_record.json   split file path and its SHA256, geometry SHA256, stride, fps, colour limits,
                      git commit, package versions (numpy, matplotlib, ffmpeg -version first line)
```

v1 deliverables are `rho` × {inner, full} for both datasets: 4 movies plus 2 timelines. MP4s go to
group storage (`/work/hdd/bgun/bench/movies/<split_id>/`), not to git. Only `timeline.png`, the stills
and the records are committed.

## Code shape (so evaluation can reuse it)

- `bench/viz/geometry.py`: `slice_coords(geo, view) -> (x, y)` and `slice_values(state, view) -> v`.
  These are pure functions with no I/O.
- `bench/viz/frame.py`: `render_frame(panels, spec) -> PIL.Image` or `fig`. `panels` is a list of
  (title, values); `spec` holds the colour limits, zoom and timeline info. The future truth / prediction
  / residual layout is the same function with 3 panels.
- CLI: `bench movie --split-file splits/<id>.json --quantity rho --zoom inner [--stride N]` and
  `bench timeline --split-file splits/<id>.json`. Defaults come from the table above; `--stride` exists
  only for quick previews, and preview outputs get a `_preview` suffix.
- Render frames in parallel with `multiprocessing.Pool` (each worker loads its own dump). Each output
  must not depend on the worker count.
- Slurm: a CPU job, 1 node, 32 cores, 4 h, on Delta (`--partition=cpu`). Don't render on login nodes.

## Acceptance checks

1. A unit test on a synthetic geometry with an axisymmetric field: the meridional left and right halves
   match within float tolerance, and the equatorial slice is rotationally uniform.
2. A field set to rho = r⁻¹ shows concentric circles in the equatorial view and circles in the
   meridional view, with no wedge or seam except at the θ crop.
3. Running twice gives byte-identical `colour_limits.json`, `timeline.csv` and stills (fixed fonts, no
   timestamps in PNG metadata: `metadata={"Software": None}`).
4. The segment colours and chunk ticks in the timeline strip match the split file's `segments` first, last and
   count for each split.
