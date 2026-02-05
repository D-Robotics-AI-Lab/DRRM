import os
import glob
import re
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, vfx

def merge_episodes(target_rel_path, merged_name="0_merged_episodes.mp4", speed=4.0):
    # Target directory path
    
    # Construct absolute path assuming script is run from project root
    # or adapt if needed
    base_dir = os.getcwd()
    target_dir = os.path.join(base_dir, target_rel_path)
    
    # Check if directory exists
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        # Try to find it if we are deeper in the structure or user provided path relative to root
        if os.path.exists(target_rel_path):
             target_dir = target_rel_path
        else:
             print("Please run this script from the project root or check the path.")
             return

    print(f"Scanning directory: {target_dir}")
    
    # Find episode files
    mp4_files = glob.glob(os.path.join(target_dir, "episode*.mp4"))
    
    if not mp4_files:
        print("No 'episode*.mp4' files found in the directory.")
        return

    # Sort files by episode number
    def get_episode_idx(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r"episode(\d+)\.mp4", basename)
        if match:
            return int(match.group(1))
        return 999999

    mp4_files.sort(key=get_episode_idx)
    
    clips = []
    print(f"Found {len(mp4_files)} videos. Processing...")

    for fpath in mp4_files:
        idx = get_episode_idx(fpath)
        print(f"Processing episode {idx}...")
        
        # Load clip
        clip = VideoFileClip(fpath)
        
        # 4x speed
        # Using v2 style effects
        clip = clip.with_effects([vfx.MultiplySpeed(speed)])
        
        # Create text label with episode number
        # Using white text on black background for visibility
        try:
            # Add spaces for padding
            label_text = f"{idx}"
            # Try with method='label' which is often cleaner for single line text
            txt = TextClip(
                text=label_text,
                font_size=32,
                size=(38,38),
                vertical_align='top',
                color='white',
                bg_color='black',
                method='label'
            )
        except TypeError:
            # Fallback for older signatures or slight API variations
            txt = TextClip(
                txt=label_text,
                fontsize=32,
                size=(38,38),
                vertical_align='top',
                color='white',
                bg_color='black'
            )
        except Exception as e:
            # If TextClip fails (e.g. no ImageMagick), print warning and skip text
            print(f"Warning: Could not create TextClip (ImageMagick installed?). Error: {e}")
            txt = None
        if txt:
            # Position at top-left with margin
            txt = txt.with_duration(clip.duration).with_position((10, 10))
            comp = CompositeVideoClip([clip, txt])
            clips.append(comp)
        else:
            clips.append(clip)

    if clips:
        print("Concatenating clips...")
        final = concatenate_videoclips(clips)
        
        output_path = os.path.join(target_dir, merged_name)
        print(f"Writing result to {output_path}...")
        
        # Use a reliable codec
        final.write_videofile(
            output_path, 
            fps=20, # Reduced fps for GIF-like speed or keep original? 
            # Original clips have fps, final clip usually inherits or defaults. 
            # Explicit fps is safer. 
            codec="libx264", 
            audio_codec="aac"
        )
        print("Done.")

if __name__ == "__main__":
    merge_episodes("simulation/robotwin2.0/eval_result/beat_block_hammer/VAR0/agilex_config/var0mix_invdit_fm_23d_1f_obs_noise-None-100/2026-01-07 01:32:31")
