# glovegen 2

Turn a high-detail positive scan into a 3D-printable, castable **two-part rigid
mold**: a block with the scanned shape hollowed out of it, split along a curved
parting surface, with alignment keys, a pour spout and vents.

Runs end to end on the real 2.6M-triangle `Hand_Child.stl` scan in **~50
seconds**, at full input resolution — no decimation of the cavity.

```
load 12.4s · direction search 3.5s · parting surface 1.7s · block−part 1.1s
split 1.9s · keys+spout+vents 5.4s · separation check 10.7s
```

```bash
pip install -e .

# analysis only: which way should the mold pull?
glovegen analyze data/samples/Hand_Child.stl --heatmap heat.ply

# build the mold
glovegen mold data/samples/Hand_Child.stl -o out/

# or use the web app
uvicorn server.app:app --reload   # then open http://127.0.0.1:8000
```

---

## How it works

### 1. Pull direction

A two-part mold pulls one half along `+d` and the other along `-d`. A face is
**not** an undercut just because its normal points against `d` — it belongs to
the other half, which releases it fine. A true undercut is geometry neither half
can reach.

Both facts come from one cheap query: cast a single ray per grid column parallel
to `d` and read the sorted crossings.

```
 column with 2 crossings          column with 4 crossings
 ───────────────────────          ───────────────────────
   ░░░ mold (pulls +d)              ░░░ mold (pulls +d)
   ▓▓▓ part                         ▓▓▓ part
   ░░░ mold (pulls -d)              ▒▒▒ mold  ← trapped fin: cast above
                                    ▓▓▓ part     and below it, so it cannot
   both blocks lift off             ░░░ mold     leave along +d *or* -d
```

So the quantity to minimise is

```
trapped fin volume = Σ (interior gap lengths) × cell area
```

which is a real volume in mm³ — literally how much mold has to be forced past
the cast — is independent of where the parting surface ends up, and costs one ray
per column. `demold.suggest_direction` sweeps a coarse hemisphere then refines
around the leaders. Near-ties (a flattish organic shape has dozens of directions
tied at ~0%) break on how complicated the parting surface will be, then on how
much material the block costs.

Sanity check that the crossing model is right: summing the *material* intervals
instead of the gaps reproduces `mesh.volume` to 0.1% on every test shape, and
exactly (708.12 cm³) on the hand.

### 2. Parting surface

The parting surface is a **height field** `z = h(x, y)` in the frame where `d` is
+Z, defined over the whole block footprint. That one choice buys a lot: a height
field is single-valued so it *cannot* self-intersect, it triangulates into a
guaranteed-manifold sheet, and extruding it upward gives a watertight
"everything above the parting surface" solid.

It is constrained, not free. In every column that passes through the part, `h` is
pinned inside the thickest slab of part material:

- through the palm, `h` runs through the middle of the hand, so the mold splits
  into a palm-side half and a back-of-hand half — the natural parting;
- approaching the silhouette the slab thins to nothing, the band collapses, and
  `h` is forced through the silhouette **exactly**. The surface follows the
  undercut boundary with no loop extraction, contouring or ear-clipping;
- between the fingers no column hits the part, so `h` is unconstrained and simply
  interpolates smoothly across the gap.

Solving it is a small QP — minimise `λ‖∇h‖² + ‖h − target‖²` subject to
`band_lo ≤ h ≤ band_hi` — which is one sparse linear solve plus a few active-set
passes. 2.0 s on the hand at a 401×141 grid.

Grid resolution affects **seam quality, not correctness**: coarsening it moves
where the seam sits on the cast, but `h` stays inside the cavity either way, so
the halves still reproduce the shape exactly.

### 3. Block, and the split

```
mold   = block − part
half_a = mold ∩ {above parting surface}
half_b = mold − {above parting surface}
```

Three booleans on the input mesh at full resolution. No offsetting, no distance
grids, no marching cubes. "Wall thickness" is just the margin between the part
and the block's outside, which the block's construction guarantees by definition.

`--block hull` uses the part's convex hull dilated by the margin instead of a
rectangular box: the same minimum-wall guarantee with **55% less material**
(1575 cm³ vs 3515 cm³ on the hand).

### 4. Mold features

Two geometric facts drive these:

**Halves always separate.** Half A ⊆ `{z > h}` and half B ⊆ `{z < h}`. Translating
A by `+t·d` keeps it inside `{z > h + t}`, so it can never meet B. Any feature
that respects the parting surface cannot break the split — which is why channels
can be cut from both halves freely.

**A groove wider inside than at its mouth traps the cast.** A channel of radius
`r` centred at `y₀` and cut at `h` leaves half A a circular segment whose widest
point is at `y₀`; if `h < y₀` that widest point is *above* the mouth and the cast
is locked in. So channels are either centred on the parting surface (each half
gets a clean half-round groove, widest exactly at the mouth) or routed along the
pull direction, where they release by construction. Never in between.

- **Alignment keys** — drafted frustums on the parting face, placed by
  farthest-point sampling over columns that miss the cavity, so they spread out
  and clear the part. The socket is cut as the key's **swept** volume along `+d`,
  not its resting shape: the parting surface is curved, so half B can hold
  material above the key's mouth within the key's own footprint, and without
  sweeping, the key jams on it.
- **Pour spout** — a funnel into the cavity's extreme along the pour axis,
  centred on the parting surface. The pour axis defaults to the part's longest
  principal axis oriented **fat-end-up**; on the hand scan that correctly puts
  the cut wrist at the top with the fingers hanging down, which is the
  orientation that traps the least air.
- **Vents** — thin channels from cavity high points out to the block surface.
  High points are found as local maxima of the cavity's ceiling along the pour
  axis, one per connected pocket (a flat ceiling is one pocket however many cells
  it spans). Each is routed along whichever of `±d` does not re-enter the cavity.

### 5. Verification

`validate.separation_report` measures rather than assumes, by sliding the halves
and intersecting:

- `opens` — can the halves come apart from each other? Guaranteed by
  construction, so a real number here means a feature broke the invariant.
- `cast_interference_mm3` — how much mold must be forced past the cast. Sampled
  along a **ladder** of travels, because a half that fouls the cast for the first
  millimetre is clear of it by the time it has moved 10% of the part's size, and
  a single far-travel probe reports a spurious zero.

On the hand: `opens: true`, cast interference **27.9 mm³ = 0.004%** of a 708 cm³
cast. The ray-based trapped-volume estimate independently says tens of mm³, so
the two methods agree. A rigid cast would technically bind on that; a flexible
silicone cast absorbs it, which is the design assumption.

#### On "watertight"

Exported STLs may report `watertight: false` while being perfectly closed.
manifold3d legitimately emits solids containing a few *position-coincident*
vertices; re-welding those on reload turns them into non-manifold edges, which
trips trimesh's strict flag. The number that matters is `boundary_edges` — zero
means no holes — so `mesh_stats` reports `closed`, `boundary_edges` and
`nonmanifold_edges` separately rather than a single misleading boolean. The
pipeline's validity gate follows the same rule: a genuinely open boundary is
never tolerated, a handful of coincident touchings is.

---

## What changed from v1, and why

| | v1 | v2 |
|---|---|---|
| Mold body | offset the part surface outward into a shell | `block − part` |
| Cost driver | SDF grid + marching cubes + boolean, `O((extent/pitch)³)` | three booleans, no grid |
| Full-scale hand run | never completed | **13 s** for the split, ~45 s end to end |
| Parting surface | extract loops, project, ear-clip; silently self-intersected | constrained height field; cannot self-intersect |
| Undercut metric | silhouette face area | trapped fin **volume** (mm³), parting-surface-independent |
| Small undercut islands | merged into neighbours, bridging holes that don't release | reported and measured; flexible-cast assumption made explicit |
| Keys / spout / vents | scoped, never built | built, with the withdrawal-sweep and groove-geometry constraints handled |
| Persistence | in-process dict, unenforced TTL, lost on restart | disk-backed store, enforced TTL, restart reaps orphans |

**The "2-part molds don't generalise to hands" conclusion did not reproduce.**
v1 concluded a hand has finger gaps that are undercuts from every direction. That
is an artifact of measuring undercut as silhouette *area*: finger gaps are only
undercuts when viewed *along* the finger splay. Measured as trapped volume, the
palm normal gives **0.008%** and the search finds it in 3 s. The 33 residual
undercut loops v1 fought with are, in this formulation, 28 mm³ of interference —
below the noise floor of a flexible cast.

The scale problem was likewise not intrinsic to the goal; it was intrinsic to the
offset-shelling approach.

## Known limits

- The mold produces a **solid positive** of the scan. Casting a hollow glove or
  liner needs a matching core, which is out of scope here.
- Residual undercuts are reported, not eliminated. There is no N-part split;
  the design assumes a flexible cast material.
- A box block around a hand-and-forearm scan is ~4.2 L of plastic. Use
  `--block hull`, or crop the scan to the hand.
- No auto-tiling to a printer bed: the hand mold's halves are ~150 × 78 × 386 mm
  and will not fit a 256 mm bed without manual splitting.
- One mold job runs at a time by default (`GLOVEGEN_WORKERS`); a full-resolution
  run peaks near 4 GB.

## Layout

```
glovegen/
  demold.py         pull-direction search, per-face undercut heatmap
  parting.py        constrained height field -> parting surface + solid
  mold.py           block, block−part, the split
  features.py       keys, pour spout, vents
  pipeline.py       orchestration + reporting
  validate.py       solid gating, separation measurement
  cli.py            glovegen analyze | mold
server/
  app.py            FastAPI: upload, heatmap, jobs, downloads
  store.py          disk-backed persistence, TTL, orphan reaping
  worker.py         job bodies, run out-of-process
  static/           three.js viewer with live heatmap
tests/              130 tests, ~6 s
```

## Configuration

Environment: `GLOVEGEN_STORE` (default `data/store`), `GLOVEGEN_TTL_HOURS` (24),
`GLOVEGEN_MAX_UPLOAD_MB` (400), `GLOVEGEN_WORKERS` (1).

Everything geometric lives in `glovegen/config.py` as dataclasses and is
accepted as a partial nested dict by the API, so
`{"block_margin": 12, "parting": {"grid": 500}}` is a valid job config.
