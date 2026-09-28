"""
Data Collection Tool — scripts/collect_data.py

Opens the webcam and lets the operator record step-by-step clips
for training/validation.

Controls:
  1-8    — Mark "starting step N" (records frames to data/raw/step_XX/)
  SPACE  — Pause/resume recording
  S      — Save a snapshot frame
  R      — Toggle continuous recording mode
  Q/ESC  — Quit

Usage:
  python scripts/collect_data.py
  python scripts/collect_data.py --camera 0 --output data/raw
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="SIH Data Collection Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("=" * 60)
    print("SIH Data Collection Tool")
    print("=" * 60)
    print("Controls:")
    print("  1-8    — Start recording for step N")
    print("  SPACE  — Pause/resume")
    print("  S      — Save snapshot")
    print("  R      — Toggle continuous recording")
    print("  Q/ESC  — Quit")
    print("=" * 60)

    current_step = 0
    recording = False
    continuous = False
    frame_count = 0
    session_id = time.strftime("%Y%m%d_%H%M%S")
    video_writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw UI overlay
        display = frame.copy()
        h, w = display.shape[:2]

        # Top bar
        cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
        if current_step > 0:
            step_text = f"Recording Step {current_step}"
            color = (0, 255, 0) if recording else (0, 200, 255)
        else:
            step_text = "Press 1-8 to select a step"
            color = (200, 200, 200)

        cv2.putText(display, step_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = f"Frames: {frame_count} | {'REC' if recording else 'PAUSED'}"
        if continuous:
            status += " | CONTINUOUS"
        cv2.putText(display, status, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show recording indicator
        if recording:
            cv2.circle(display, (w - 20, 25), 10, (0, 0, 255), -1)

        cv2.imshow("SIH Data Collection", display)

        # Save frame if recording
        if recording and current_step > 0:
            step_dir = output_dir / f"step_{current_step:02d}" / session_id
            step_dir.mkdir(parents=True, exist_ok=True)

            frame_path = step_dir / f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1

            if video_writer is not None:
                video_writer.write(frame)

        # Handle input
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break

        elif key == ord(' '):  # SPACE — toggle recording
            recording = not recording
            print(f"Recording {'started' if recording else 'paused'}")

        elif key == ord('s'):  # S — snapshot
            snap_dir = output_dir / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snap_dir / f"snap_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(snap_path), frame)
            print(f"Snapshot saved: {snap_path}")

        elif key == ord('r'):  # R — toggle continuous recording
            continuous = not continuous
            if continuous and current_step > 0:
                recording = True
                # Start video recording
                vid_dir = output_dir / f"step_{current_step:02d}"
                vid_dir.mkdir(parents=True, exist_ok=True)
                vid_path = vid_dir / f"video_{session_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(vid_path), fourcc, 15.0, (args.width, args.height)
                )
                print(f"Continuous recording to: {vid_path}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("Continuous recording stopped")

        elif ord('1') <= key <= ord('8'):  # 1-8 — select step
            new_step = key - ord('0')
            if new_step != current_step:
                # Stop previous video recording
                if video_writer:
                    video_writer.release()
                    video_writer = None

                current_step = new_step
                frame_count = 0
                recording = True
                session_id = time.strftime("%Y%m%d_%H%M%S")
                print(f"\n→ Step {current_step} selected. Recording started.")
                print(f"  Frames will be saved to: {output_dir}/step_{current_step:02d}/")

    # Cleanup
    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Collected data in: {output_dir}")


if __name__ == "__main__":
    main()
