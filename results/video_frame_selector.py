import cv2
import tempfile
from pathlib import Path
from src.pipeline.recognize import recognize
from src.pipeline.prepare_nose import prepare_nose

def recognize_from_video(video_path):
    """
    Extracts frames from a video, finds the one with the highest nose confidence,
    and runs the biometric recognition on it.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Could not open video"}

    # Sample 1 frame every 0.5 seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / 2) if fps > 0 else 15
    
    best_score = -1.0
    best_frame = None
    frame_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                temp_path = Path(temp_dir) / f"frame_{frame_count}.jpg"
                cv2.imwrite(str(temp_path), frame)
                
                # Check if it's a valid nose
                prep = prepare_nose(image_path=temp_path)
                if prep["status"] == "accepted":
                    if prep["validator_score"] > best_score:
                        best_score = prep["validator_score"]
                        best_frame_path = temp_path
                        
            frame_count += 1
        cap.release()

        if best_frame and best_frame_path.exists():
            # Run full recognition on the best frame found
            result = recognize(best_frame_path)
            return result
        else:
            return {"status": "not_a_nose", "message": "No valid dog nose found in video"}

# Example usage:
# result = recognize_from_video("whatsapp_video.mp4")
