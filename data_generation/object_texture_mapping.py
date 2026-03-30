#!/usr/bin/env python3
"""
Object-to-Texture Category Mapping

Maps HUMOTO objects to appropriate texture categories based on their real-world material.
Objects can be assigned to multiple texture categories and one will be randomly selected.
Objects not assigned textures will use solid colors, with realistic colors for specific items.
"""

import os
import random

# Texture base directory
TEXTURES_BASE = "./textures/general/organized"

# Available texture categories and their directories
TEXTURE_CATEGORIES = {
    "wood": [
        "Wood_001",
        "Wood035_4K-JPG",
        "Wood047_4K-JPG",
        "Wood060_4K-JPG",
        "Wood063_4K-JPG",
        "Wood064_4K-JPG",
        "Wood075_4K-JPG",
        "Wood086_4K-JPG",
        "Bamboo001C_2K-JPG",
    ],
    "metal": [
        "Metal008_2K-JPG",
        "Metal011_2K-JPG",
        "Metal012_2K-JPG",
        "Metal030_4K-JPG",
        "Metal046A_2K-JPG",
        "Metal050A_4K-JPG",
        "Metal058A_2K-JPG",
        "PaintedMetal004_2K-JPG",
    ],
    "wicker": ["Wicker004_4K-JPG"],
    "fabric": [
        "fabric_001",
        "fabric_004",
        "Fabric051_2K-JPG",
        "Fabric076_2K-JPG",
    ],
    "leather": [
        "leather_001",
        "leather_002",
        "leather_003",
        "Leather009_2K-JPG",
        "Leather036B_4K-JPG",
    ],
}

# Object material mapping
# Objects that should use textures (key = object name, value = list of possible texture categories)
# Multiple categories mean the object can realistically be any of those materials
TEXTURED_OBJECTS = {
    # Baskets - wicker
    "basket": ["wicker"],
    "basket_90": ["wicker"],
    "woven_basket": ["wicker"],

    # Wood/Metal dual options
    "cutting_board": ["wood"],
    "dining_chair": ["wood", "metal", "fabric", "leather"],  # Chairs can be many materials
    "drawer": ["wood", "metal"],
    "guitar": ["wood"],
    "kitchen_counter": ["wood", "metal"],
    "kitchen_counter_small": ["wood", "metal"],
    "low_chair": ["wood", "metal", "fabric", "leather"],
    "shelf": ["wood", "metal"],
    "side_table": ["wood", "metal"],
    "step_stool": ["wood", "metal"],
    "table": ["wood", "metal"],
    "table_lamp": ["metal", "fabric"],  # Base metal, shade can be fabric
    "ukelele": ["wood"],
    "utility_cart": ["metal"],
    "working_chair": ["fabric", "leather", "metal"],  # Office chairs
    "bed": ["wood", "metal", "fabric"],  # Frame and headboard

    # Metal objects
    "bathroomsink": ["metal"],
    "can": ["metal"],
    "clothes_hanger": ["metal", "wood"],
    "clothes_rack": ["metal", "wood"],
    "floor_lamp": ["metal"],
    "fork": ["metal"],
    "frying_pan": ["metal"],
    "hammer": ["metal", "wood"],  # Handle can be wood
    "knife": ["metal"],
    "laptop": ["metal"],
    "laptop_bottom": ["metal"],
    "laptop_top": ["metal"],
    "peeler": ["metal"],
    "screwdriver": ["metal"],
    "shower_squeegee": ["metal"],
    "sink": ["metal"],
    "soap_dispenser": ["metal"],
    "soap_dispenser_body": ["metal"],
    "soap_dispenser_top": ["metal"],
    "soup_ladle": ["metal"],
    "spatula": ["metal"],
    "spoon": ["metal"],
    "tap": ["metal"],
    "turner": ["metal"],
    "vacuum_flask": ["metal"],
    "vacuum_flask_body": ["metal"],
    "vacuum_flask_cap": ["metal"],
    "whisk": ["metal"],
    "wok_turner": ["metal"],

    # Wicker/Basket objects
    "draw_organizer_tray": ["wicker", "wood", "fabric"],
    "organizer_medium": ["wicker", "fabric"],
    "organizer_small": ["wicker", "fabric"],
    "tray": ["wicker", "wood", "metal"],
    "wash_tub": ["fabric", "metal"],
    "notebook": ["leather", "fabric"],
}

# Realistic colors for specific objects (RGB values 0-1 range)
REALISTIC_COLORS = {
    # Fruits - realistic colors
    "mango": [
        (0.95, 0.65, 0.15),  # Orange-yellow
        (0.98, 0.75, 0.20),  # Golden yellow
        (0.90, 0.45, 0.15),  # Orange-red
        (0.92, 0.70, 0.25),  # Yellow-orange
    ],
    "orange": [
        (1.0, 0.55, 0.0),    # Bright orange
        (1.0, 0.60, 0.10),   # Orange
        (0.95, 0.50, 0.05),  # Deep orange
        (1.0, 0.65, 0.15),   # Light orange
    ],

    # Flowers - varied colors
    "flower": [
        (0.95, 0.20, 0.40),  # Red
        (0.95, 0.75, 0.85),  # Pink
        (0.90, 0.85, 0.30),  # Yellow
        (0.85, 0.40, 0.85),  # Purple
        (0.20, 0.50, 0.90),  # Blue
        (0.30, 0.70, 0.45),  # Green
    ],

    # Plastic/ceramic items - varied but sensible (more saturated)
    "deep_plate": [
        (0.90, 0.80, 0.65),  # Beige (more saturated)
        (0.65, 0.85, 0.95),  # Light blue (more saturated)
        (0.95, 0.20, 0.20),  # Red
        (0.25, 0.95, 0.40),  # Green
    ],
    "mixing_bowl": [
        (0.95, 0.20, 0.20),  # Red (more saturated)
        (0.65, 0.85, 0.95),  # Light blue (more saturated)
        (0.95, 0.90, 0.60),  # Cream (more saturated)
        (0.25, 0.95, 0.40),  # Green
    ],
    "mug": [
        (0.95, 0.20, 0.20),  # Red (more saturated)
        (0.20, 0.50, 0.90),  # Blue (more saturated)
        (0.20, 0.20, 0.20),  # Black
        (0.85, 0.65, 0.50),  # Brown (more saturated)
        (0.25, 0.95, 0.40),  # Green
    ],
    "plastic_bowl": [
        (0.98, 0.25, 0.25),  # Red (more saturated)
        (0.25, 0.65, 0.98),  # Blue (more saturated)
        (0.25, 0.95, 0.40),  # Green (more saturated)
        (0.98, 0.95, 0.25),  # Yellow (more saturated)
    ],
    "plastic_bowl_stacked": [
        (0.98, 0.25, 0.25),  # Red (more saturated)
        (0.25, 0.65, 0.98),  # Blue (more saturated)
        (0.25, 0.95, 0.40),  # Green (more saturated)
    ],
    "serving_bowl": [
        (0.90, 0.80, 0.65),  # Beige (more saturated)
        (0.65, 0.85, 0.95),  # Light blue (more saturated)
        (0.95, 0.20, 0.20),  # Red
        (0.25, 0.95, 0.40),  # Green
    ],
    "side_plate": [
        (0.90, 0.80, 0.65),  # Beige (more saturated)
        (0.65, 0.85, 0.95),  # Light blue (more saturated)
        (0.95, 0.20, 0.20),  # Red
        (0.25, 0.95, 0.40),  # Green
    ],
    "vase": [
        (0.20, 0.50, 0.80),  # Blue (more saturated)
        (0.85, 0.70, 0.55),  # Tan (more saturated)
        (0.30, 0.70, 0.45),  # Green (more saturated)
        (0.95, 0.20, 0.20),  # Red
    ],

    # Tables - NO WHITE, darker/richer colors only
    "table": [
        (0.45, 0.30, 0.20),  # Dark brown/wood
        (0.35, 0.25, 0.20),  # Darker brown
        (0.25, 0.25, 0.30),  # Dark gray
        (0.30, 0.20, 0.15),  # Deep brown
    ],
    "side_table": [
        (0.45, 0.30, 0.20),  # Dark brown/wood
        (0.35, 0.25, 0.20),  # Darker brown
        (0.25, 0.25, 0.30),  # Dark gray
    ],
    "kitchen_counter": [
        (0.35, 0.35, 0.40),  # Dark gray
        (0.25, 0.25, 0.30),  # Darker gray
        (0.40, 0.30, 0.25),  # Brown gray
    ],
    "kitchen_counter_small": [
        (0.35, 0.35, 0.40),  # Dark gray
        (0.25, 0.25, 0.30),  # Darker gray
        (0.40, 0.30, 0.25),  # Brown gray
    ],

    # Bins/trash - typically darker
    "larger_bin": [
        (0.25, 0.25, 0.25),  # Black
        (0.30, 0.35, 0.40),  # Dark gray
        (0.35, 0.40, 0.45),  # Gray
    ],
    "trash": [
        (0.25, 0.25, 0.25),  # Black
        (0.30, 0.35, 0.40),  # Dark gray
    ],
    "trash_can": [
        (0.25, 0.25, 0.25),  # Black
        (0.30, 0.35, 0.40),  # Dark gray
        (0.60, 0.60, 0.60),  # Medium gray
        (0.20, 0.50, 0.90),  # Blue
    ],

    # Other items
    "lint_roller": [
        (0.40, 0.60, 0.80),  # Blue
        (0.60, 0.60, 0.60),  # Medium gray
        (0.95, 0.20, 0.20),  # Red
    ],
    "pen": [
        (0.20, 0.30, 0.60),  # Blue
        (0.15, 0.15, 0.15),  # Black
        (0.80, 0.20, 0.20),  # Red
    ],
    "phone": [
        (0.10, 0.10, 0.10),  # Black
        (0.30, 0.35, 0.40),  # Gray
        (0.75, 0.65, 0.55),  # Gold
        (0.20, 0.50, 0.90),  # Blue
    ],
    "usb": [
        (0.15, 0.15, 0.15),  # Black
        (0.60, 0.60, 0.60),  # Medium gray
        (0.80, 0.30, 0.30),  # Red
        (0.20, 0.50, 0.90),  # Blue
    ],
}

# Default random colors for objects not specified above (bright, varied colors)
# More saturated colors, no orange (conflicts with Sophie's shirt)
DEFAULT_COLOR_POOL = [
    (0.95, 0.15, 0.15),  # Bright Red (more saturated)
    (0.15, 0.65, 0.95),  # Bright Blue (more saturated)
    (0.15, 0.95, 0.35),  # Bright Green (more saturated)
    (0.98, 0.90, 0.15),  # Bright Yellow (more saturated)
    (0.85, 0.15, 0.85),  # Bright Purple (more saturated)
    (0.15, 0.95, 0.95),  # Bright Cyan (more saturated)
    (0.98, 0.45, 0.65),  # Bright Pink (more saturated)
    (0.50, 0.15, 0.85),  # Bright Violet (more saturated)
]

# Orange colors - only used for Sophie when explicitly needed
ORANGE_COLORS = [
    (0.95, 0.55, 0.15),  # Bright Orange
    (0.98, 0.65, 0.20),  # Light Orange
]


def get_texture_for_object(obj_name):
    """
    Get appropriate texture path for an object, or None if it should use color.

    Args:
        obj_name: Name of the object

    Returns:
        str: Path to texture directory, or None if object should use solid color
    """
    # Check if object should use texture
    if obj_name not in TEXTURED_OBJECTS:
        return None

    # Get possible texture categories for this object
    possible_categories = TEXTURED_OBJECTS[obj_name]

    # Randomly select one category from the possibilities
    category = random.choice(possible_categories)

    # Get available textures in this category
    if category not in TEXTURE_CATEGORIES:
        print(f"  Warning: Unknown texture category '{category}' for {obj_name}")
        return None

    available_textures = TEXTURE_CATEGORIES[category]

    # Select random texture from category
    texture_name = random.choice(available_textures)
    texture_path = os.path.join(TEXTURES_BASE, texture_name)

    # Verify texture exists
    if not os.path.exists(texture_path):
        print(f"  Warning: Texture path not found: {texture_path}")
        return None

    return texture_path


def get_color_for_object(obj_name, character_name=None):
    """
    Get realistic color for an object.

    Args:
        obj_name: Name of the object
        character_name: Name of character ('sophie' or 'remy'), used to avoid color conflicts

    Returns:
        tuple: RGB color tuple (0-1 range)
    """
    # Check if object has specific realistic colors
    if obj_name in REALISTIC_COLORS:
        return random.choice(REALISTIC_COLORS[obj_name])

    # For Sophie scenarios, avoid orange (her shirt is orange)
    # Exception: orange and mango objects should still be orange
    if character_name and character_name.lower() == 'sophie':
        if obj_name.lower() in ['orange', 'mango']:
            # These objects should still be orange
            return random.choice(ORANGE_COLORS) if obj_name.lower() == 'orange' else random.choice(REALISTIC_COLORS.get('mango', ORANGE_COLORS))
        else:
            # Use default color pool (no orange)
            return random.choice(DEFAULT_COLOR_POOL)

    # Otherwise use default color pool (no orange for any character to be safe)
    return random.choice(DEFAULT_COLOR_POOL)


def should_use_texture(obj_name):
    """
    Determine if an object should use texture or solid color.

    Args:
        obj_name: Name of the object

    Returns:
        bool: True if object should use texture, False for solid color
    """
    return obj_name in TEXTURED_OBJECTS


def get_texture_categories(obj_name):
    """
    Get the possible texture categories for an object.

    Args:
        obj_name: Name of the object

    Returns:
        list: List of texture category names, or None if not a textured object
    """
    return TEXTURED_OBJECTS.get(obj_name, None)


def print_texture_summary():
    """Print summary of texture assignments."""
    print("\n" + "="*60)
    print("Object Texture Assignment Summary")
    print("="*60)

    # Count by category
    category_counts = {}
    multi_material_objects = []

    for obj, categories in TEXTURED_OBJECTS.items():
        if len(categories) > 1:
            multi_material_objects.append((obj, categories))
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\nTextured Objects: {len(TEXTURED_OBJECTS)}")
    print("\nTexture Categories Used:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} object assignments")

    print(f"\nMulti-Material Objects: {len(multi_material_objects)}")
    for obj, cats in sorted(multi_material_objects):
        print(f"  {obj}: {', '.join(cats)}")

    print(f"\nObjects with Realistic Colors: {len(REALISTIC_COLORS)}")
    for obj in sorted(REALISTIC_COLORS.keys()):
        print(f"  {obj}: {len(REALISTIC_COLORS[obj])} color options")

    print("="*60)


def list_unmapped_objects(all_object_names):
    """
    List objects that don't have texture assignments (will use colors).

    Args:
        all_object_names: List of all object names in dataset
    """
    mapped = set(TEXTURED_OBJECTS.keys())
    unmapped = [obj for obj in all_object_names if obj not in mapped]

    if unmapped:
        print("\n" + "="*60)
        print("Objects Using Colors (not textured)")
        print("="*60)
        for obj in sorted(unmapped):
            has_realistic = "✓ realistic colors" if obj in REALISTIC_COLORS else "random colors"
            print(f"  {obj}: {has_realistic}")
        print("="*60)

    return unmapped


if __name__ == "__main__":
    # Print summary when run directly
    print_texture_summary()

    # Test a few objects
    print("\nExample Texture/Color Assignments:")
    test_objects = ["basket", "dining_chair", "fork", "mixing_bowl", "guitar", "mango", "orange", "table"]

    print("\n5 random samples per object:")
    for obj in test_objects:
        print(f"\n  {obj}:")
        for i in range(5):
            if should_use_texture(obj):
                texture_path = get_texture_for_object(obj)
                if texture_path:
                    category = os.path.basename(texture_path)
                    print(f"    {i+1}. TEXTURE → {category}")
            else:
                color = get_color_for_object(obj)
                print(f"    {i+1}. COLOR → RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
