# An adapter for each kind of sound

Date: 2026-10-08. The choices marked "user" were made by the person the tool was built for.
Nothing here has been run; see "Known and not known".

## Why

Stable Audio 3 makes its own idea of a drone or a texture. The library holds the user's idea
of one. An adapter (a LoRA) is a small file trained on a set of sounds that pulls the model
toward them. Stable Audio 3's checkout already has a trainer that runs on this Mac and a
`--lora` option for generating.

User: one adapter per kind of sound (Drone, Ambience, Texture and so on), not one per show and
not one for everything. Its own guidance agrees: "One coherent style per dataset (one artist,
one genre, one SFX category). Mixed bags train into mush." A second use follows (user): take
a texture, generate a variation of it with the Drone adapter, and get a drone made from it.

User: sounds should also carry character words: spacy or close, intense or soft, dark or
bright. These become the words an adapter is trained with, and filters on the page.

## What it does

### Using an adapter

When at least one adapter is ready, the generate panel shows an **Adapter** row: None, then
each adapter by name. It works with every Make choice and when generating from text.

- An adapter belongs to the model it was trained on. Choosing one switches Model to match.
- The adapter's word (for the Drone adapter, "drone") is put at the start of the prompt
  unless the prompt already has it, so an empty prompt works.
- **Adapter strength** under More, 0 to 2, 1 by default.
- A clip's description says which adapter and strength made it.

### Making an adapter

Four commands, because the steps are hours apart and the second and third make the Mac work
hard. Each says what it will do and how long it expects to take before it starts.

1. `audio-embed adapter gather KIND` makes a collection named `Training KIND` holding every
   file of that kind that is the user's own (`own recording` and `own show work`).
   `--from "sample pack"` and the like widen it. It says how many files and minutes it found;
   Stable Audio 3's trainer wants 10 minutes at the least and 30 or more to do well.
   The collection is then an ordinary one: play it on the page, take out what does not
   belong, keep other files into it. That is how the user decides what an adapter learns.
   If the collection is already there the command leaves it alone.
2. `audio-embed adapter prepare NAME --model sfx` turns the collection into what the trainer
   reads. Each file is copied through ffmpeg as a 44.1 kHz stereo WAV (the first 10 minutes
   of a longer one) with a caption beside it, Stable Audio 3 turns the copies into its
   compressed form, and the copies are deleted. Online-only files are skipped and counted.
3. `audio-embed adapter train NAME --model sfx [--steps 10000]` runs the trainer, showing its
   progress, and keeps the Mac awake while it does. It saves a checkpoint and a few demo
   clips every 1000 steps. Stopping it loses nothing; running it again carries on from the
   last checkpoint. The first run for a model downloads its training weights (0.9 GB for a
   small model, 2.9 GB for medium).
4. `audio-embed adapter use NAME --model sfx [--step 6000]` chooses the checkpoint the panel
   uses, the last one by default. The demos are there to choose by: more steps is not
   better, and the trainer's makers put the best point "around 10k".

`audio-embed adapter` alone lists every adapter and how far along it is.

### Character words

Each file gets up to three words, one from each pair: **spacy** or **close**, **intense** or
**soft**, **dark** or **bright**. They are worked out the way the kind of sound is: CLAP
compares the file with a description of each side. A file gets a word only when it is clearly
on one side; otherwise it gets none for that pair, because a wrong word in a caption teaches
the wrong thing.

- They show on the page as a **character** filter group, with no new code on the page.
- Like the kinds, they are not stored and can be changed in a file next to the index
  (`data/characters.json`).
- The caption of a training file is its character words, such as `dark, spacy`. The trainer
  shuffles them, sometimes drops some, and puts the adapter's word in front four times out
  of five, so the adapter learns its kind and leaves the meaning of the other words to the
  model.

## Where things are written

Everything is under `data/adapters/NAME/MODEL/`:

- `set/`: the WAV copies and captions, only while preparing.
- `latents/`: what the trainer reads. Can be made again by `prepare`.
- `runs/`: checkpoints and demos. Hours of work, and the largest part: a checkpoint is 50 to
  200 MB by Stable Audio 3's own account, so ten of them can be 2 GB.
- `adapter.json`: the adapter's word, its model, which checkpoint is in use, and how many
  files and minutes it was trained on.

`gather` adds one collection under `data/collections/`. Nothing is written to the library,
the kept folder or the render drive.

## Parts

- `classify.py`: the three pairs and `characters()`, next to the kinds. `server.py` adds the
  words to each file's labels where it adds the kind.
- `adapters.py` (new): everything about making and finding adapters. Which files a kind has,
  captions, the trainer's prompt settings, the three Stable Audio 3 command lines
  (pre-encode, train, demos), where each thing lives, which adapters are ready.
- `generate.py`: `Ask` gains `adapter` and `adapter_strength`; `command` adds
  `--lora FILE strength=S` and the word; the story and `describe` gain the adapter.
- `server.py`: the generate view's JSON lists the adapters that are ready; `POST
  /api/generate` takes an adapter **by name**, checked against that list. No path ever comes
  from the page.
- `cli.py`: `audio-embed adapter` and its four steps.
- `web/index.html`: the Adapter row and Adapter strength.

## Rules

None change, and one decides the design: the library is read with ffmpeg and nothing else.
That is why `prepare` makes copies instead of pointing Stable Audio 3 at the collection's
shortcuts. The copies also fix three things its reader would get wrong: it cannot open the
`.m4a` files, it resamples by drawing straight lines between samples, and it keeps only the
first two channels of a wider file.

`data/adapters/` joins the list of things in `data/` that cannot be rebuilt.

## Licence

Stable Audio 3's weights are under the Stability AI Community License. Its page says (read
2026-10-08): free for "any business making under USD $1M annually of revenue"; "you own
outputs generated from the Core Models or Derivative Works (such as fine-tunes)"; adapters
are allowed. Whether a bought sample pack may be used to train on is for that pack's licence
to say, which is why `gather` takes only the user's own files unless told otherwise.

## Known and not known

Known, from Stable Audio 3's code and documents:

- The trainer runs on Apple Silicon with no other software, and an adapter trained there
  loads with `--lora` in every mode the generator uses.
- Its makers report 1.65 steps a second for a small model on an M4 Pro, on 24-second
  excerpts. The default excerpt is 120 seconds, so a real run is several times slower.
  **10,000 steps is a night, not the 20 minutes guessed earlier.**
- Medium can be trained on a Mac ("16 GB+ recommended"); no speed is given.

Not known until the Mac may make noise:

- How many files and minutes each kind has among the user's own. The kinds are not stored, so
  counting needs CLAP.
- How fast training is on the M2 Max, for a small model and for medium.
- Whether an adapter changes the sound enough to hear, and whether a variation of a texture
  made with the Drone adapter comes out as a drone.
- How often the character words are right, and where "clearly on one side" should be drawn.

So the work starts with a trial, not with the commands: count the kinds, listen to the
clearest files on each side of each pair, then train one adapter (Drone, on the small sfx
model) by hand and compare clips made with and without it from the same seeds. What follows
is built only if that is worth keeping.

## Not in this version

- Training from the page, or watching a run there.
- Captions from file names, embedded tags or a captioning model.
- Using two adapters at once (Stable Audio 3 can).
- Taking in an adapter that was trained elsewhere.
- Clearing out old checkpoints.

## Testing

Unit tests cover the character words (a file clearly on one side gets the word, one in
between gets none), which files `gather` picks, captions and prompt settings, the three
command lines, which adapters count as ready, and the adapter's part in `command`, in request
checking and in a clip's description. With stand-ins for ffmpeg's output and for Stable
Audio 3, they cover `prepare` and `use` from a collection to an adapter the panel lists.
Training itself is checked by hand, against a scratch index and a scratch `data/`.
