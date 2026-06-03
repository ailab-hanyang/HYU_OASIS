"""
Scene Context — Prompts & Schema

28 context items, VLM system/user prompts, camera name constants.
"""

# ============================================================
#  Ring Camera Names (AV2 Sensor Dataset)
# ============================================================
CAMERA_NAMES = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]

# Cameras used for the 'ego' multi-view query
EGO_CAMERA_NAMES = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
]


# ============================================================
#  28 Context Items — keys by category
# ============================================================
CONTEXT_SCHEMA = {
    # Annotates whether each item is visible in each of the 7 cameras in boolean(true/false)
    "infra": [
        "bus_stop",
        "parking_lot",
        "railroad_tracks",
        "fence",
        "red_painted_lane",
        "construction_zone",
        "gas_station",
    ],
    
    "weather": [
        "rain",
        "snow",
        "cloudy",
        "clear_day",
    ],

    "time_of_day": [
        "dusk_dawn",
        "daylight",
    ],

    # Whether the ego itself is on / in / over / facing the element.
    "ego": [
        # Road surface (ego is traveling on / over)
        "bridge",
        "brick_street",
        "pothole",
        "filled_pothole",
        "storm_grate",
        "road_damage",
        "streetcar_tracks",
        # Environment containment (ego is inside)
        "roundabout",
        "school_zone",
        "construction_zone",
        "speed_limit_zone",
        # Lighting context affecting the ego
        "shadow_of_building",
        # Traffic signals governing the ego's path
        "green_light",
        "yellow_light",
        "broken_traffic_light",
    ],
}


# ============================================================
#  VLM Prompts — Per-camera query (infra / weather / time_of_day)
# ============================================================

SYSTEM_PROMPT = """You are an autonomous driving scene analyzer.
Given camera images from the ego vehicle, you determine which context elements are visible in the image.

Judgment policy — be BALANCED:
- Mark TRUE when there is recognizable visual evidence of the element, including partial or distant instances.
- Do NOT mark TRUE based only on loosely associated cues.
- If the element is clearly absent or the evidence points to a different category, mark FALSE.

Output policy:
- Respond with a single valid JSON object only.
- No explanation, no markdown, no code fences.
- Use exactly the keys specified in the user message.
- Every value must be boolean (true or false)."""


USER_PROMPT = """Analyze this driving camera image and determine the presence of each context element below.

This image is one of several ring-camera views around the ego vehicle (the
specific camera is named after this list). Judge each element only from what
is visible in THIS image. For the weather items, rely on the sky when it is
visible in this view. For time-of-day, use the overall ambient lighting,
which is present in every view.

## Infrastructure & Road Attributes

- bus_stop:
  Evidence of a dedicated bus stop is visible — a bus stop sign (bus icon or
  'BUS STOP' text), a transit shelter with bus/transit branding, a marked
  bus bay with 'BUS ONLY' text, or a bus actively stopped at the curb
  picking up or dropping off passengers.
  A plain bench, a generic shelter without branding, or a bus simply
  driving past does NOT qualify.

- parking_lot:
  An off-street parking area, or its entrance/driveway, is visible —
  a paved area beside, behind, or in front of a building (commercial
  lot, dealership, rental agency, apartment lot) where multiple vehicles
  are parked off the roadway, a driveway/curb-cut leading into such a
  lot (with the lot or its parked vehicles visible beyond), or a
  parking garage entrance. Painted stall lines or 'PARKING' signage
  also qualify but are NOT required.

- railroad_tracks:
  Railroad tracks, a railroad crossing sign, or railroad crossing gates
  are visible.

- fence:
  A fence (chain-link, construction fence, wooden fence, or metal railing)
  is visible along or near the road.

- red_painted_lane:
  A road lane whose surface is painted solid red (or reddish-orange) is
  visible — typically used to designate transit-only, bus-only, or
  bike-priority lanes. The red coloring covers the full lane width or a
  substantial portion of it.
  A red-painted crosswalk, a small red marking, red curb paint, or a
  red-tinted brick surface does NOT qualify.

- construction_zone:
  An active construction zone is visible somewhere in the camera image
  — multiple construction elements together (signs, barriers, cones,
  drums, equipment, or workers) or a clearly marked work area. A single
  isolated cone does NOT qualify.
  Note: this is a per-camera visibility flag, used for matching other
  tracks against the construction environment via camera-frustum
  partitioning. It does NOT require the ego to be inside the zone — for
  the ego-inside judgement, the separate 'ego.construction_zone' item
  is used.

- gas_station:
  A gas/fuel station is visible — fuel pumps under a canopy, a station
  brand sign (e.g. Shell, Chevron, BP, Exxon, Mobil, 76, Arco, Sunoco),
  a fuel price display board, or the station's forecourt with parked
  vehicles refueling. The entrance/exit driveway of such a station also
  qualifies if the pumps or canopy are visible beyond.
  An EV charging station alone, a generic convenience store, or an auto
  repair shop without fuel pumps does NOT qualify.

## Weather

- rain:
  Rain is evident — wet/glossy road, raindrops on the lens, active wipers,
  water spray, or visible rainfall.

- snow:
  Snow is evident — snow on the ground, snow-covered surfaces, falling
  snow, or plowed snow piles along the road.

- cloudy:
  The sky is visibly overcast or cloudy without direct sunshine.

- clear_day:
  The sky is clear with visible blue sky or direct sunshine.

## Time of Day

- dusk_dawn:
  Low-sun twilight conditions are visible. Mark TRUE if any one of:
  a warm orange/pink tint along the horizon (even a thin band, even
  with a blue upper sky); long raking shadows from a low sun; or an
  overall dim scene with weak low-angle light. A uniformly bright
  midday sky with no warm horizon band and no long shadows does NOT
  qualify.

- daylight:
  The scene is in daytime conditions (not dusk, dawn, or night). Overcast
  daylight still counts.

## Output Schema

Respond with exactly this JSON format (single line, no whitespace, no line breaks):
{"infra": {"bus_stop": false, "parking_lot": false, "railroad_tracks": false, "fence": false, "red_painted_lane": false, "construction_zone": false, "gas_station": false}, "weather": {"rain": false, "snow": false, "cloudy": false, "clear_day": false}, "time_of_day": {"dusk_dawn": false, "daylight": false}}"""


# ============================================================
#  VLM Prompts — Ego query (multi-view, all 5 cameras considered)
# ============================================================

EGO_SYSTEM_PROMPT = """You are an autonomous driving scene analyzer.
You receive five camera images captured at the same timestamp from the
ego vehicle, covering the surroundings in this order:
  - front-center: forward path the ego is driving into
  - front-left, front-right: lateral surroundings on each side at the
    current position
  - rear-left, rear-right: rearward lateral surroundings (what the ego
    has just passed through, still partially visible behind)
Your job is to determine, from the EGO VEHICLE'S OWN PERSPECTIVE,
whether each listed element applies to the ego itself — not whether it
merely exists somewhere in the scene.

How to read the five views (consider ALL of them together — no single
view is the anchor):
- Front-center shows the ego's CURRENT path and what is directly ahead.
- Front-left and front-right show what is flanking the ego right now on
  each side (bridge rails, construction barriers along the lane,
  building shadow boundary at the ego's side, roundabout island on a
  side).
- Rear-left and rear-right show what the ego has just passed through,
  and confirm whether the ego is still inside the same segment (same
  structure visible behind, signage the ego just passed continues to
  govern).
- Combine evidence across all five views before deciding. For
  traffic-signal items (green_light, yellow_light, broken_traffic_light),
  only the front-center view physically captures the forward signal
  governing the ego — lateral and rear views simply do not contain that
  information.

Judgment policy — be STRICT about onset:
- Mark TRUE only when the ego is UNAMBIGUOUSLY ALREADY on / over /
  inside the element at this moment. The element must be physically
  flanking, surrounding, or directly under the ego — not merely visible
  ahead.
- "About to enter", "approaching", "entrance just ahead", "the structure
  is visible a short distance in front", or seeing the feature in the
  distance all map to FALSE. Onset-but-not-yet cases are NOT enough.
- "Just entered" / "currently passing through" qualify as TRUE ONLY
  when the surrounding evidence is unambiguous — the relevant structures
  already flank the ego, the surface change is already under the ego,
  or governing signage has already been passed and is currently in
  effect.
- Mark FALSE when the element is merely visible elsewhere in the scene
  without containing or supporting the ego (e.g. a bridge ahead the ego
  is approaching but not yet on; a school zone sign on a side road; a
  green light controlling a cross street).
- If the evidence is clearly absent or only loosely associated, mark
  FALSE.
- When in doubt between "about to enter" and "just entered", prefer
  FALSE.

Output policy:
- Respond with a single valid JSON object only.
- No explanation, no markdown, no code fences.
- Use exactly the keys specified in the user message.
- Every value must be boolean (true or false)."""


EGO_USER_PROMPT = """Determine whether each ego-centered element below applies to the ego vehicle in this timestamp.

## Road Surface (the ego is traveling ON or OVER it)

Strict onset rule for this section: mark TRUE only when the feature is
physically under the ego right now, or the ego's wheels are about to
contact it within a very short distance directly ahead in the same lane.
"Approaching from far away", "the feature is visible somewhere ahead",
or "the ego will soon reach it" all map to FALSE.
Exception: where an individual item below states its own looser rule
(e.g. storm_grate, which also accepts a grate set into the gutter beside
the ego), follow that item's rule instead of this section default.

- bridge:
  Decide this from the SIDE views first, not the dramatic structure
  ahead. Necessary condition: a bridge rail / parapet / steel beam /
  suspension cable / pylon physically FLANKS the ego at the immediate
  edge of its lane in the front-left and/or front-right view (lower-
  middle of the frame), NOT merely centered ahead in the front-center
  view. If neither side shows such structure flanking the ego now,
  mark FALSE.
  Strengthen the judgement with either: the surface under the ego is a
  recognizable bridge deck (steel grid, expansion joints, or distinct
  deck concrete, unlike the asphalt approach); OR open space (river,
  valley, water, or a road below) is visible past the road edge beside
  the ego. On a long bridge, the rear-left / rear-right views still
  showing the same rails trailing behind confirm the ego is still on it.
  Mark FALSE when: the bridge is only visible ahead / centered while the
  ego's lane is still ordinary asphalt; the ego is approaching or "about
  to enter"; the ego is passing UNDER an overpass (structure overhead,
  not flanking at road level); or a distant bridge is seen across a
  valley / water from a normal road. When in doubt, prefer FALSE.

- brick_street:
  The road surface the ego is currently traveling on is paved with
  bricks, cobblestones, or stone pavers — a clearly non-asphalt
  textured surface with a visible block/brick pattern in the ego's
  lane in the front view.
  Brick or cobblestone limited to sidewalks, crosswalks, or decorative
  borders (with the ego's lane itself being asphalt) does NOT qualify.

- pothole:
  An unfilled pothole — a broken-out hole in the road surface, dark
  inside, with depth and irregular jagged edges — lies in the ego's
  path, directly in front of the ego in the same lane within a short
  distance (the ego is about to pass over it, is currently over it, or
  has just passed over it). A pothole in another lane, on the shoulder,
  or far ahead in a different part of the road does NOT qualify.
  A smooth patched repair (use filled_pothole) does NOT qualify.

- filled_pothole:
  A patched / filled-in pothole — a smooth, even, paved-over repair
  patch whose color or texture clearly differs from the surrounding
  road surface (typically a darker fresh-asphalt rectangle or a tar
  seal with no remaining hole) — lies in the ego's path, directly in
  front of the ego in the same lane within a short distance (about to
  pass over, currently over, or just passed over). A patch in another
  lane, on the shoulder, or a normal expansion joint / construction
  seam does NOT qualify. A still-open hole (use pothole) does NOT
  qualify.

- storm_grate:
  A storm drain grate or drainage grate is visible ON or immediately
  BESIDE the ego's own roadway within a short distance — either in the
  ego's lane, or set into the curb gutter / shoulder running directly
  alongside or just ahead of the ego. Unlike the other road-surface
  items in this section, the ego need NOT be about to drive directly
  over it: a grate in the gutter at the ego's side of the road qualifies.
  A grate on a far sidewalk, across the intersection, on the opposite
  side of a divided road, or on a different road does NOT qualify.

- road_damage:
  The road surface the ego is currently traveling on shows significant
  deterioration in the ego's lane — visible cracks (longitudinal,
  transverse, or alligator cracking), broken or crumbling pavement,
  severe wear, or large patched repair areas in the ego's path.
  Construction cones or signs marking damage in the ego's lane also
  qualify when the damage is implied by their placement on the ego's
  driving path. Damage limited to other lanes, the shoulder, or the
  sidewalk does NOT qualify. A single small pothole alone (use pothole)
  or a clean storm grate does NOT qualify.

- streetcar_tracks:
  Parallel steel rails embedded in the road surface run in the ego's
  direction of travel and are within or immediately beside the ego's
  lane (the ego is on or directly alongside the tracks). Tracks visible
  only on a cross street, far ahead at an intersection the ego is
  approaching but not yet crossing, do NOT qualify.

## Environment Containment (the ego is INSIDE)

Strict onset rule for this section: mark TRUE only when the ego is
UNAMBIGUOUSLY ALREADY inside the environment — the entry point has been
crossed and the surrounding cues (structure, signage, lane geometry)
already apply to the ego's current position. "About to enter",
"entrance / sign just ahead but not yet reached", or "the environment
is visible a short distance in front" all map to FALSE. When in doubt
between "approaching" and "just inside", prefer FALSE.

- roundabout:
  The ego is currently inside a roundabout or traffic circle. Mark TRUE
  when the ego's lane curves continuously to one side (typically left in
  right-hand traffic) around a central feature. A central island —
  planted, paved, monument, or just a circular kerbed/curbed area in the
  middle of the intersection — visible in ANY camera view (in particular
  the front-left or side/rear-left views as the ego rounds the circle)
  is sufficient. The island may be small, obscured by other vehicles,
  far from the ego, or only partially in frame — do NOT require it to
  be fully visible or centered ahead. Radial exit legs branching off,
  or circulating arrows / yield-on-entry markings on the ego's path,
  are additional supporting cues but not required when the curving path
  + a central island feature are clear from the multi-view context.
  Approaching from outside with the entry yield line ahead but not yet
  crossed does NOT qualify.

- school_zone:
  The ego is currently within an active or signed school zone. Mark TRUE
  when ANY of the following is satisfied in ANY of the ego's multi-view
  cameras: a school zone sign, a pentagonal school-crossing sign,
  'SCHOOL' pavement text, a yellow school-zone flashing beacon, OR a
  clearly identifiable school bus (yellow body with 'SCHOOL BUS'
  lettering or a stop-arm) on or beside the ego's road. A school
  building visible alongside a normal road without zone signage or a
  driving school sign does NOT qualify.

- construction_zone:
  The ego is inside or actively passing through a construction work
  area — multiple construction elements (signs, barriers, cones, drums,
  equipment, or workers) line or surround the ego's lane, or the ego's
  lane has been shifted / narrowed due to the work. A single isolated
  cone, or a construction site visible far off in the distance that the
  ego is not yet entering, does NOT qualify.

- speed_limit_zone:
  The ego is currently inside a posted speed-limit segment — a speed
  limit sign (e.g. 'SPEED LIMIT 25', 'SPEED LIMIT 35') is visible ahead
  applying to the ego's direction of travel, OR such a sign was clearly
  recently passed (still partially visible at the image edges or just
  behind), establishing the segment the ego is in. A speed limit sign
  on the opposite side of a divided road, applying to the opposite
  direction, does NOT qualify on its own.

## Lighting Affecting the Ego

- shadow_of_building:
  The ego is inside shading caused by surrounding buildings. Two
  qualifying patterns — either is sufficient:
  (a) A discrete building shadow falls across the ego's road, with the
      ego's lane surface visibly in shadow and a sunlit area visible
      elsewhere in the scene (the boundary may be soft and need not be
      sharp).
  (b) URBAN-CANYON shading: the ego is flanked by buildings on at least
      two sides and the overall lighting at the ego's position feels
      subdued / softly shaded compared to fully open-sky stretches of
      road — even when the scene is NOT very dark and there is NO clear
      light/dark boundary anywhere. A uniformly lightly-shaded urban
      block surrounded by buildings counts.
  The sky being bright vs the ego's road being slightly muted is a
  valid cue. Do NOT require strong contrast or deep darkness.
  Tree shade alone, underpass shade, or a cloudy overcast scene with
  no nearby buildings do NOT qualify (use the 'cloudy' per-camera item
  for overcast).

## Traffic Signals Governing the Ego

For these items, the question is NOT 'is any traffic light visible?' —
it is 'does a traffic light directly governing the ego's lane / path
of travel show this state, clearly visible from the ego's front view?'.
A signal controlling a cross street, a left-turn-only bay the ego is
not in, or a far-side intersection beyond the next one does NOT qualify.
A pedestrian signal head does NOT qualify.
Exception: broken_traffic_light is judged per signal head (see its rule
below) — a dead head facing the ego still qualifies even when another
head at the same intersection is working normally.

- green_light:
  In the ego's front view, the vehicular traffic light directly governing
  the ego's current lane / path of travel displays an illuminated round
  green lens. A green directional arrow on an overhead sign, a pedestrian
  walk signal, and a green light on a cross street all do NOT qualify.

- yellow_light:
  In the ego's front view, the vehicular traffic light directly governing
  the ego's current lane / path of travel has its middle lamp actively
  illuminated amber/yellow. The fluorescent yellow back plate (frame)
  without a lit lamp does NOT qualify. A yellow signal on a cross street
  does NOT qualify.

- broken_traffic_light:
  Judge each signal head facing the ego's direction of travel on its
  OWN — a head can be broken even when a neighboring head at the same
  intersection shows a normal color, so do NOT let a working green/yellow
  elsewhere in the view suppress this judgement.
  Mark TRUE when a full round-lens (red-yellow-green ball) head whose
  LENS SIDE faces the ego is non-functional: every lens dark / unlit (a
  dead head) even though a companion head is lit, OR the head is flashing
  abnormally (e.g. all-flashing red as a temporary 4-way stop), OR it is
  visibly damaged or knocked askew.
  The following do NOT qualify: the flat dark BACK of a signal head aimed
  at the cross street (housing facing away, no lenses toward the ego); a
  dedicated turn-ARROW head that is simply unlit while its companion ball
  signal works normally (this is normal operation); a pedestrian signal
  head; or a signal beyond the next intersection.

## Output Schema

Respond with exactly this JSON format (single line, no whitespace, no line breaks):
{"bridge": false, "brick_street": false, "pothole": false, "filled_pothole": false, "storm_grate": false, "road_damage": false, "streetcar_tracks": false, "roundabout": false, "school_zone": false, "construction_zone": false, "speed_limit_zone": false, "shadow_of_building": false, "green_light": false, "yellow_light": false, "broken_traffic_light": false}"""