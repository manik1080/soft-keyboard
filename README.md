# Smart Touch Keyboard

A desktop learning prototype for a probabilistic soft keyboard. It demonstrates:

- Per-key bivariate Gaussian touch distributions
- Mahalanobis-distance-based spatial scoring
- A small prefix-based language prior
- Context-dependent Gaussian scaling
- Per-key touch calibration
- Live visualization of Gaussian contours, 2σ rectangles and posterior candidates

Mobile keyboard input is subject to errors that are generally attributed to “fat finger typing”.
I built this because i was interested in learning how keyboards work on smartphones. As soon as I heard that the keyboards are actually dynamic, i fell into a rabbit hole. I referenced google research page about their keyboards and how they use overlapping 2D Gaussian probability field combined with a Finite-State Transducer (FST) and Neural Language Model (NN-LM) to dynamically shift effective hitboxes based on typing context. And so I wanted to build something similar myself.

Future work includes rendering this on a smartphone screen rather than desktop. I used tkinter here on desktop for ease of building the ui, it's lightweight and good enough for this prototype... and because i'm comfortable mostly with python. I plan on working on building a platform independent app version so it can be loaded onto a handheld device for more accurate usage.

## Project structure

```text
smart_touch_keyboard_project/
├── app.py       # Application controller, event handling and entry point
├── ui.py        # Tkinter widgets, canvas drawing and display updates
├── model.py     # Keyboard geometry, probability model, language prior and calibration
└── README.md    # Setup and architecture notes
```

## Run

Python 3.9+ is recommended. Tkinter is part of most standard Python installations.

Open a terminal in this folder and run:

```bash
python app.py
```

On Windows, this also works:

```powershell
py app.py
```

No third-party packages are required.

## Responsibilities

### `model.py`

Owns the keyboard geometry and the mathematical model:

- Key rectangles and key centres
- Per-key Gaussian means and covariance parameters
- Mahalanobis distance and log-likelihood scoring
- Prefix-based next-character priors
- Context-dependent covariance scaling
- Calibration sample collection and parameter fitting
- Gaussian ellipse coordinates for visualization

### `ui.py`

Contains only the Tkinter presentation layer:

- Window layout, panels, buttons and checkboxes
- Keyboard canvas and key drawing
- Gaussian contours and bounding rectangles
- Candidate list and model inspector
- Updating the decoded text and calibration status

It receives callbacks from `app.py`; it does not decide which character to type.

### `app.py`

Acts as the application controller:

- Creates the model and UI
- Connects UI events to model operations
- Manages typed text, undo and Shift state
- Runs the calibration workflow
- Refreshes the UI after state changes
- Starts the Tkinter event loop

## Important interpretation notes

- Grey keycaps are the physical key rectangles.
- Teal ellipses are 1σ Gaussian contours.
- Orange dashed rectangles enclose the 2σ Gaussian contours; they are **not** hard hitboxes.
- The blue outline marks the most likely next letter from the language prior.
- Gold and teal keycap highlights show the first, second and third posterior candidates for the current pointer preview.
