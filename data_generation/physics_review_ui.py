#!/usr/bin/env python3
"""
Streamlit UI for manual physics review.

This UI allows you to review rendered first frames and select which objects
should NOT have physics applied (i.e., which objects should remain static).

Usage:
    streamlit run physics_review_ui.py
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image

# Configuration
FRAMES_DIR = Path("./physics_review_frames")
METADATA_FILE = FRAMES_DIR / "sequences_metadata.json"
CONFIG_FILE = Path("./physics_config.json")


def load_metadata():
    """Load sequence metadata."""
    if not METADATA_FILE.exists():
        st.error(f"Metadata file not found: {METADATA_FILE}")
        st.info("Run `python generate_review_frames.py` first to generate frames and metadata")
        return []

    with open(METADATA_FILE, 'r') as f:
        return json.load(f)


def load_config():
    """Load existing physics configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save physics configuration."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    st.set_page_config(
        page_title="Physics Review Tool",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Physics Review Tool")
    st.markdown("""
    Review each scenario and select which objects should **remain static** (no physics applied).

    By default, physics is applied to ALL objects when the human is removed.
    Select objects that should NOT move (e.g., tables, shelves, etc.).
    """)

    # Load data
    sequences = load_metadata()
    if not sequences:
        st.stop()

    config = load_config()

    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Sidebar navigation
    st.sidebar.header("Navigation")

    # Progress
    reviewed_count = sum(1 for seq in sequences if seq['name'] in config)
    st.sidebar.metric("Progress", f"{reviewed_count}/{len(sequences)}")
    st.sidebar.progress(reviewed_count / len(sequences) if sequences else 0)

    # Jump to sequence
    st.sidebar.subheader("Jump to Sequence")

    # Filter options
    filter_option = st.sidebar.radio(
        "Show",
        ["All", "Reviewed", "Not Reviewed"]
    )

    # Apply filter
    filtered_sequences = sequences
    if filter_option == "Reviewed":
        filtered_sequences = [s for s in sequences if s['name'] in config]
    elif filter_option == "Not Reviewed":
        filtered_sequences = [s for s in sequences if s['name'] not in config]

    # Search
    search_term = st.sidebar.text_input("Search by name")
    if search_term:
        filtered_sequences = [s for s in filtered_sequences
                             if search_term.lower() in s['name'].lower()]

    # Sequence selector
    if filtered_sequences:
        sequence_names = [s['name'] for s in filtered_sequences]
        selected_name = st.sidebar.selectbox(
            "Select Sequence",
            sequence_names,
            index=min(st.session_state.current_index, len(sequence_names) - 1)
        )

        # Update index based on selection
        st.session_state.current_index = sequence_names.index(selected_name)
        current_seq = filtered_sequences[st.session_state.current_index]
    else:
        st.warning(f"No sequences found matching criteria")
        st.stop()

    # Navigation buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("⬅️ Prev"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            st.rerun()
    with col2:
        if st.button("➡️ Next"):
            st.session_state.current_index = min(len(filtered_sequences) - 1,
                                                  st.session_state.current_index + 1)
            st.rerun()
    with col3:
        if st.button("⏭️ Skip to\nUnreviewed"):
            # Find next unreviewed
            for i in range(st.session_state.current_index + 1, len(sequences)):
                if sequences[i]['name'] not in config:
                    # Find position in filtered list
                    for j, seq in enumerate(filtered_sequences):
                        if seq['name'] == sequences[i]['name']:
                            st.session_state.current_index = j
                            st.rerun()
                            break
                    break

    # Main content
    st.header(f"Scenario: {current_seq['name']}")

    # Display image
    image_path = FRAMES_DIR / f"{current_seq['name']}.png"
    if image_path.exists():
        image = Image.open(image_path)
        # Double size from previous (960px width)
        st.image(image, width=960, caption="First frame with human")
    else:
        st.warning(f"Image not found: {image_path}")
        st.info("Run `python generate_review_frames.py` to generate this frame")

    # Object selection
    st.subheader("Objects in Scene")

    if not current_seq['objects']:
        st.info("No objects in this scene")
    else:
        st.markdown("""
        **👇 Check the boxes for objects that should HAVE PHYSICS (will fall/move):**
        - ✅ Check: `mug`, `bowl`, `plate`, `utensils`, `bottle` (will fall when human removed)
        - ⬜ Leave unchecked: `table`, `shelf`, `counter`, `cabinet`, `chair` (stay static - DEFAULT)
        """)

        # Get current configuration
        current_config = config.get(current_seq['name'], {
            'physics_objects': [],
            'notes': ''
        })

        # Show checkboxes for each object
        st.write("**Select ONLY objects that should have PHYSICS (will move/fall):**")
        st.info("💡 By default, all objects are STATIC. Only check boxes for objects that should move!")

        physics_objects = []

        # Create columns for better layout (3 columns)
        num_cols = 3
        cols = st.columns(num_cols)

        for idx, obj in enumerate(current_seq['objects']):
            col_idx = idx % num_cols
            with cols[col_idx]:
                # Checkbox for each object - checked means PHYSICS
                has_physics = obj in current_config.get('physics_objects', [])
                if st.checkbox(f"💥 {obj}", value=has_physics, key=f"obj_{current_seq['name']}_{obj}"):
                    physics_objects.append(obj)

        # Calculate static objects (all others - DEFAULT)
        static_objects = [obj for obj in current_seq['objects']
                          if obj not in physics_objects]

        # Display summary
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**⚡ Physics Objects ({len(physics_objects)}):**")
            if physics_objects:
                for obj in physics_objects:
                    st.markdown(f"- 💥 {obj}")
            else:
                st.markdown("*None - all objects will remain static*")

        with col2:
            st.success(f"**📌 Static Objects ({len(static_objects)}):**")
            if static_objects:
                for obj in static_objects:
                    st.markdown(f"- {obj}")
            else:
                st.markdown("*None - all objects will have physics*")

        # Notes field
        notes = st.text_area(
            "Notes (optional)",
            value=current_config.get('notes', ''),
            placeholder="Any special notes about this scenario..."
        )

        # Save button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Configuration", type="primary"):
                config[current_seq['name']] = {
                    'static_objects': static_objects,
                    'physics_objects': physics_objects,
                    'all_objects': current_seq['objects'],
                    'notes': notes
                }
                save_config(config)
                st.success("✓ Saved!")

                # Auto-advance to next scenario
                if st.session_state.current_index < len(filtered_sequences) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
                else:
                    st.balloons()
                    st.info("🎉 That was the last scenario!")

        with col2:
            if st.button("🗑️ Clear Configuration"):
                if current_seq['name'] in config:
                    del config[current_seq['name']]
                    save_config(config)
                    st.success("✓ Cleared!")
                    st.rerun()

    # Show existing configuration status
    if current_seq['name'] in config:
        st.info("✓ This scenario has been reviewed and configured")
    else:
        st.warning("⚠️ This scenario has not been configured yet")

    # Statistics sidebar
    st.sidebar.subheader("Statistics")
    st.sidebar.text(f"Total Sequences: {len(sequences)}")
    st.sidebar.text(f"Reviewed: {reviewed_count}")
    st.sidebar.text(f"Remaining: {len(sequences) - reviewed_count}")

    # Export option
    st.sidebar.subheader("Export")
    if st.sidebar.button("📥 Download Config"):
        st.sidebar.download_button(
            label="Download physics_config.json",
            data=json.dumps(config, indent=2),
            file_name="physics_config.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
