# Redo a part, loop, make longer

Date: 2026-10-08. The choices marked "user" were made by the person the tool was built for.

## Why

Generate from a sample makes a whole new clip that drifts from the sample. Three common wants
are narrower than that (user: all three):

- one moment in a sound is wrong and the rest is right;
- an ambience or drone has to repeat under a scene without a click or a jump;
- a sound is right but too short.

Stable Audio 3 can regenerate one stretch of a clip while it hears the rest ("inpainting",
`--inpaint-range "START,END"`). The checkout the generator already runs has it, for all three
models. All three wants are that one call with the stretch placed differently.

## What it does

The panel of the generate view gets a **Make** row with four choices. The first is what the
view does today.

- **Variations**: unchanged. Prompt and How far.
- **Redo a part**: drag on the sample's waveform to mark the part. Only that part is made
  anew; the prompt says what it should be and may be empty.
- **Loop**: the end and the start are made anew so that the clip repeats without a seam.
  **Join** says how much is replaced: Short, Medium or Long (1, 2 or 4 seconds, half of it at
  each end).
- **Longer**: new material follows on from the sample. **Add** says how much: 10, 30 or 60
  seconds. A marker on the waveform says where the new material takes over. It starts at the
  end of the file and can be pulled back, because most samples end in a fade and there is
  nothing to follow on from in silence.

How far is hidden for the three new choices: the part is made from nothing but the prompt and
what surrounds it, so there is no distance to set. Count, Model, Avoid and Prompt strength
work as they do now. Each run still makes several clips from different seeds, listed, played
and kept as now. A loop's row plays on repeat, so the join is heard.

Each clip row gets **Work from this**. It makes that clip the sample of the view, so a
variation can be looped, or a clip made from text made longer, before anything is kept. The
view says which clip it is working from and offers the way back to the file.

### Which passage is used

Stable Audio 3 is given at most 60 seconds, as now (120 when a length is asked for).

- **Redo a part**: the passage is placed around the marked part, with as much before as
  after where the file allows. At least 1 second must be left untouched, or there is nothing
  to match. Length under More does not apply; the passage is at most 60 seconds.
- **Loop**: the passage is the one Variations would use. Length under More sets how long the
  loop is. The clip must be at least 2 seconds, and the join is at most half of it.
- **Longer**: the passage ends at the marker and reaches back up to 60 seconds. Passage and
  new material together are at most 120 seconds.

From a long file the result is the passage with the change in it, not the whole file.

### What stays untouched

Stable Audio 3 keeps the rest of a clip exactly only in its own compressed form; the whole
clip still passes through its codec. So the tool takes only the new part from what comes
back, and joins it into the passage it sent with a 30 millisecond crossfade at each edge.
Outside the part and those fades, a result is sample for sample the passage that was cut
from the file (44.1 kHz, 16 bit, stereo, as every clip is now).

This also decides how a loop is made. The passage is turned so that its end and start meet in
the middle, the join is made anew there, the new part is joined into the turned passage, and
the result is turned back. The place where the clip repeats then lies inside one unbroken
stretch of new audio, and the place where it was turned back is the file's own audio, which
was continuous there to begin with.

Stable Audio 3 works in steps of 4096 samples (93 ms). A part's edges are moved to the
nearest step, and a passage that will be turned is trimmed to a whole number of steps, so the
tool and the model agree on where the part is.

## Parts

- `splice.py` (new): the arithmetic on samples, with no model in it. Reading and writing the
  16-bit stereo WAV, moving a time to the nearest step, turning a clip, joining a new part
  into an original with crossfades.
- `generate.py`:
  - `Ask` gains `make` (`variations`, `part`, `loop`, `longer`), `span` (start and end in the
    file, for a part), `marker` (where new material takes over, for longer), `join` and
    `add`. `ask_from` checks them.
  - Choosing the passage for each `make`, and where the part lies inside it.
  - `command` still passes the passage as `--init-audio`, with `--inpaint-range` where
    Variations passes the noise level. For longer,
    `--seconds` is the passage plus what is added; Stable Audio 3 pads the passage with
    silence itself.
  - `_make` turns the passage first for a loop, and after each clip joins the new part in
    (and turns a loop back) before the clip goes into the cache.
  - A run can start from a clip in the cache. Its clips are filed under the same library
    file (or under text) as the clip they came from.
  - The story of a clip gains `make`, the part in clip time, `join`, `add`, and the
    description of the clip it was made from, if any. `describe` tells the whole chain, such
    as: `Generated from "Pad 04". Then 0:12 to 0:15 regenerated. Then made to loop (2 s
    join).` A story without `make` is a variation, so clips already in the cache still read.
- `server.py`: `POST /api/generate` also takes `make`, `span`, `join`, `add` and `clip` (a
  clip id, checked as it is everywhere else). `GET /api/generated` takes `clip` too and then
  describes that clip as the sample. A clip's JSON says what was made, for its row.
- `web/index.html`: the Make row; the part and the marker on the sample's waveform, dragged
  with the mouse; Join and Add; Work from this; repeat playback for a loop. The choice of
  Make and the clip being worked from are in the URL hash like the rest of the view.

## Rules

None change. The library is read only through ffmpeg, to cut the passage. The arithmetic
happens on files in a scratch folder. Clips go to the cache and are kept exactly as now, and
a clip is still reached by its id alone.

## Not yet heard

None of this has been listened to. It is known from Stable Audio 3's code that the calls
exist and what they do, not how the edges of a part sound, whether a join made this way
loops well, or how convincing a continuation is, and it may differ between the three models.
The first step of the work is therefore to make these by hand from a few library files, in a
scratch folder, for the user to hear. If a kind does not hold up it is left out, and the
others do not depend on it.

## Not in this version

- Adding before the start of a sample.
- Putting a redone part back into a whole long file.
- How far for a part (Stable Audio 3 can start a part from the old audio instead of from
  nothing).
- Loops cut to a tempo or to bars, and finding a good loop point automatically.
- Showing where a join or a new part lies on a made clip's own waveform.

## Testing

Unit tests cover the arithmetic (a turn there and back gives the same samples; outside the
part and its fades a joined clip equals the original exactly, and inside it equals the new
audio), the passage chosen for each kind, the command line for each kind, request checking,
and the description of a chain. With a stand-in for Stable Audio 3 that writes a known
signal, they cover the whole path from a request to a clip in the cache for each kind, and a
run started from a clip. The page and the real model are checked by hand against a scratch
index and a scratch kept folder, including that a loop repeats in the browser without a gap.
