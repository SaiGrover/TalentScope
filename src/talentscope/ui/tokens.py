"""Design tokens.

The direction is adapted from the Data Vortex Round 4 dashboard (dark canvas,
opaque cards, one card shape, tile strips with an accent rule, an eyebrow
heading per section, interpretation under every chart, a pipeline rail as the
navigation) and re-keyed for HR analytics: a cooler ink-and-teal canvas, a
teal / clay pair that always means *stayed* / *left*, and a sans pairing of
Sora for display and Inter for text.

Every categorical colour below was run through the dataviz palette validator
against both the canvas and the card surface (lightness band, chroma floor,
CVD separation, normal-vision floor and 3:1 contrast all pass). The two
ordinal ramps pass the ordinal checks.
"""

# surfaces
BG = "#0b1115"
PANEL = "#121a20"
PANEL_2 = "#18222a"
BORDER = "rgba(170,196,206,.14)"
BORDER_STRONG = "rgba(170,196,206,.28)"
GRID = "rgba(170,196,206,.10)"

# text (WCAG contrast on PANEL: 15.0, 7.4, 4.9)
TXT = "#e8eef1"
MUT = "#9aabb5"
DIM = "#7a8b96"

# categorical order (fixed, never cycled)
TEAL = "#2fa697"
CLAY = "#d9634a"
BLUE = "#4f86d8"
AMBER = "#b5842a"
VIOLET = "#8b6fd0"
GREEN = "#6aa35f"
CATEGORICAL = [TEAL, CLAY, BLUE, AMBER, VIOLET, GREEN]

# meaning-bearing colours
STAYED = TEAL
LEFT = CLAY
OUTCOME = {"Stayed": STAYED, "No": STAYED, "Left": LEFT, "Yes": LEFT, 0: STAYED, 1: LEFT}
NEUTRAL = "#5f6f79"          # de-emphasis gray for context marks
HIGHLIGHT = "#e9c46a"        # text-only accent for the brand wordmark

# status (reserved meaning, always shown with an icon and a word)
STATUS = {"pass": "#4fbf8f", "warn": "#d9a441", "fail": "#e5604d"}
STATUS_ICON = {"pass": ":material/check_circle:", "warn": ":material/warning:", "fail": ":material/cancel:"}

# strategy identity for model-comparison charts: three validated hues (all-pairs pass for
# scatter use) plus a light neutral for the no-rebalancing reference; each also has its own
# marker symbol so identity never rests on colour alone
STRATEGY_COLORS = {"none": "#8d9ca5", "class_weight": BLUE, "smotenc": TEAL, "random_oversample": AMBER}
STRATEGY_SYMBOLS = {"none": "circle", "class_weight": "square", "smotenc": "triangle-up", "random_oversample": "x"}
MODEL_SHORT = {"Dummy (prior)": "Dummy", "Logistic regression": "Logistic reg.", "Decision tree": "Decision tree",
               "Random forest": "Random forest", "Support vector machine": "SVM", "XGBoost": "XGBoost",
               "LightGBM": "LightGBM"}

# ordinal ramps (validated, light -> dark)
CLAY_RAMP = ["#f3c3b6", "#eb9c86", "#e07a61", "#c95e45", "#a64a35"]
TEAL_RAMP = ["#a9e3d9", "#6fcab9", "#36a898", "#22796d"]

# sequential scales for heatmaps (one hue, dark -> bright on a dark surface)
SEQ_CLAY = [[0, "#1a2329"], [0.25, "#4a2c27"], [0.5, "#86402f"], [0.75, "#c95e45"], [1, "#f3a58f"]]
SEQ_TEAL = [[0, "#141f25"], [0.25, "#173b3a"], [0.5, "#1f6b62"], [0.75, "#2fa697"], [1, "#9fe3d6"]]
# diverging: blue <- neutral gray -> clay
DIVERGING = [[0, "#4f86d8"], [0.25, "#3b5778"], [0.5, "#2a343b"], [0.75, "#86402f"], [1, "#d9634a"]]

FONT_DISPLAY = "Sora, Inter, 'Segoe UI', system-ui, sans-serif"
FONT_TEXT = "Inter, 'Segoe UI', system-ui, sans-serif"
