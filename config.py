from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "data.yaml"
DATA_DIR = BASE_DIR / "advanced_data"
VALID_IMAGES = DATA_DIR / "valid" / "images"
TEST_IMAGES = DATA_DIR / "test" / "images"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "skin_pro_medium_v1" / "weights" / "best.pt"
HISTORY_DB = BASE_DIR / "scan_history.db"

DISCLAIMER = (
    "SkinScan AI is for informational purposes only and is not medical advice. "
    "Always consult a licensed dermatologist for diagnosis and treatment."
)

CLASS_NAMES = [
    'Pimples', 'blackhead', 'conglobata', 'crystanlline', 'cystic',
    'folliculitis', 'keloid', 'milium', 'papular', 'purulent',
]

RECOMMENDATIONS = {
    'Pimples':      {'AM': 'Salicylic Acid Cleanser',               'PM': 'Benzoyl Peroxide 5% Spot Treatment'},
    'blackhead':    {'AM': 'BHA Liquid Exfoliant',                  'PM': 'Double Cleanse (Oil + Water)'},
    'conglobata':   {'AM': 'Gentle Hydrating Cleanser',             'PM': 'URGENT: Requires Prescription (Isotretinoin)'},
    'crystanlline': {'AM': 'Centella Asiatica Soothing Mist',       'PM': 'Ceramide Barrier Repair Cream'},
    'cystic':       {'AM': 'Anti-inflammatory Serum (Niacinamide)', 'PM': 'Adapalene 0.1% Gel + Deep Hydration'},
    'folliculitis': {'AM': 'Antibacterial Wash (Benzoyl Peroxide)', 'PM': 'Warm Compress + Mupirocin (if prescribed)'},
    'keloid':       {'AM': 'Silicone Gel / Sheet',                  'PM': 'Consult Professional for Corticosteroid info'},
    'milium':       {'AM': 'Mild Lactic Acid Exfoliation',          'PM': 'Do Not Squeeze - Professional Extraction Only'},
    'papular':      {'AM': 'Azelaic Acid 10% Suspension',           'PM': 'Zinc-based Soothing Cream'},
    'purulent':     {'AM': 'Hydrocolloid Pimple Patch',             'PM': 'Gentle Non-foaming Cleanser (Avoid physical scrubs)'},
}

SEVERITY_WEIGHTS = {
    'milium': 1, 'blackhead': 1, 'crystanlline': 1,
    'Pimples': 2, 'papular': 2,
    'purulent': 3, 'folliculitis': 3,
    'cystic': 4, 'keloid': 4,
    'conglobata': 5,
}

INGREDIENT_CONFLICTS = [
    {
        'trigger': lambda c: len(c & {'Pimples', 'blackhead', 'milium'}) >= 2,
        'warning': 'AM: Multiple acid exfoliants (Salicylic Acid, BHA, Lactic Acid) recommended. Alternate days to avoid over-exfoliation.',
    },
    {
        'trigger': lambda c: bool(c & {'Pimples', 'folliculitis'}) and 'cystic' in c,
        'warning': 'PM: Benzoyl Peroxide + Adapalene both recommended. Apply 10 minutes apart to minimize irritation.',
    },
    {
        'trigger': lambda c: 'conglobata' in c,
        'warning': 'WARNING: Conglobata is a severe condition. OTC products are insufficient — consult a dermatologist urgently.',
    },
    {
        'trigger': lambda c: 'keloid' in c and bool(c & {'Pimples', 'purulent', 'folliculitis'}),
        'warning': 'Keloid risk: Avoid Benzoyl Peroxide directly on keloid tissue. Consult a dermatologist for treatment.',
    },
]


def compute_severity(unique_classes: list) -> tuple:
    score = sum(SEVERITY_WEIGHTS.get(c, 1) for c in unique_classes)
    if score == 0:    label = 'Clear'
    elif score <= 2:  label = 'Mild'
    elif score <= 6:  label = 'Moderate'
    elif score <= 12: label = 'Severe'
    else:             label = 'Critical'
    return score, label


def check_conflicts(detected_conditions: list) -> list:
    conds = set(detected_conditions)
    return [rule['warning'] for rule in INGREDIENT_CONFLICTS if rule['trigger'](conds)]