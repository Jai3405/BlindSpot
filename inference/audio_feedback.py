"""
Audio Feedback System
Provides voice alerts and navigation guidance for blind users.
Uses pyttsx3 for offline text-to-speech (no internet required).
"""

import pyttsx3
import threading
import queue
import time
from typing import List
from inference.spatial_analyzer import SpatialObject


class AudioFeedback:
    """Text-to-speech audio feedback system."""

    def __init__(self,
                 rate=180,  # Words per minute
                 volume=1.0,  # 0.0 to 1.0
                 voice_id=None):  # System voice ID
        """
        Initialize audio feedback system.

        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice_id: System voice ID (None = default)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Set voice if specified
        if voice_id is not None:
            self.engine.setProperty('voice', voice_id)

        # Audio queue for non-blocking speech
        self.audio_queue = queue.Queue()
        self.is_speaking = False

        # Start audio thread
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

        # Alert cooldown (prevent spam)
        self.last_alert_time = {}
        self.alert_cooldown = 2.0  # Seconds between same alerts

        print("✓ Audio feedback system initialized")

    def _audio_worker(self):
        """Background worker for audio playback."""
        while True:
            try:
                text = self.audio_queue.get(timeout=0.1)
                if text is not None:
                    self.is_speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.is_speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio error: {e}")
                self.is_speaking = False

    def speak(self, text, priority=False):
        """
        Speak text.

        Args:
            text: Text to speak
            priority: If True, clear queue and speak immediately
        """
        if priority:
            # Clear queue for priority messages
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

        self.audio_queue.put(text)

    def announce_objects(self, spatial_objects: List[SpatialObject], max_objects=3):
        """
        Announce detected objects with distances.

        Args:
            spatial_objects: List of detected objects
            max_objects: Maximum number of objects to announce
        """
        if not spatial_objects:
            return

        # Announce top priority objects
        announcements = []
        for obj in spatial_objects[:max_objects]:
            # Format distance
            if obj.distance < 1.0:
                dist_str = "very close"
            elif obj.distance < 2.0:
                dist_str = f"{obj.distance:.1f} meters"
            else:
                dist_str = f"{int(obj.distance)} meters"

            # Create announcement
            announcement = f"{obj.class_name} {obj.position}, {dist_str}"
            announcements.append(announcement)

        # Speak combined announcement
        if announcements:
            text = ". ".join(announcements)
            self.speak(text)

    def announce_critical_alert(self, spatial_object: SpatialObject):
        """
        Announce critical obstacle (priority speech).

        Args:
            spatial_object: Critical obstacle
        """
        # Check cooldown
        alert_key = f"{spatial_object.class_name}_{spatial_object.position}"
        current_time = time.time()

        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.alert_cooldown:
                return  # Skip to avoid spam

        self.last_alert_time[alert_key] = current_time

        # Create urgent alert
        if spatial_object.distance < 1.0:
            text = f"Warning! {spatial_object.class_name} directly ahead!"
        else:
            text = f"Caution! {spatial_object.class_name} {spatial_object.position} at {spatial_object.distance:.1f} meters"

        self.speak(text, priority=True)

    def announce_navigation(self, navigation_hints):
        """
        Announce navigation guidance.

        Args:
            navigation_hints: Dictionary from SpatialAnalyzer
        """
        # Critical alerts first
        if navigation_hints['critical_alerts']:
            alert = navigation_hints['critical_alerts'][0]
            self.speak(f"Stop! {alert}", priority=True)
            return

        # Obstacles ahead
        if navigation_hints['obstacles_ahead']:
            obstacle = navigation_hints['obstacles_ahead'][0]
            self.speak(f"Obstacle ahead: {obstacle}")

        # Recommendations
        if navigation_hints['recommendations']:
            recommendation = navigation_hints['recommendations'][0]
            if "clear" not in recommendation.lower():
                self.speak(recommendation)

    def announce_safe_direction(self, direction):
        """
        Announce safe direction to move.

        Args:
            direction: 'left', 'center', or 'right'
        """
        if direction == 'center':
            text = "Path ahead is clear"
        else:
            text = f"Safe to move {direction}"

        self.speak(text)

    def announce_environment_summary(self, spatial_objects: List[SpatialObject]):
        """
        Provide a brief summary of the environment.

        Args:
            spatial_objects: All detected objects
        """
        if not spatial_objects:
            self.speak("No obstacles detected. Path is clear.")
            return

        # Count objects by position
        left_count = sum(1 for obj in spatial_objects if obj.position == 'left')
        center_count = sum(1 for obj in spatial_objects if obj.position == 'center')
        right_count = sum(1 for obj in spatial_objects if obj.position == 'right')

        # Count critical obstacles
        critical_count = sum(1 for obj in spatial_objects if obj.priority == 1)

        # Build summary
        if critical_count > 0:
            summary = f"{critical_count} critical obstacle"
            if critical_count > 1:
                summary += "s"
            summary += " detected. "
        else:
            summary = ""

        # Add position info
        positions = []
        if left_count > 0:
            positions.append(f"{left_count} on left")
        if center_count > 0:
            positions.append(f"{center_count} ahead")
        if right_count > 0:
            positions.append(f"{right_count} on right")

        if positions:
            summary += ", ".join(positions)

        self.speak(summary)

    def stop(self):
        """Stop all audio playback."""
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Stop engine
        self.engine.stop()

    def get_available_voices(self):
        """
        Get list of available system voices.

        Returns:
            voices: List of voice info dictionaries
        """
        voices = self.engine.getProperty('voices')
        voice_list = []

        for voice in voices:
            voice_list.append({
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender if hasattr(voice, 'gender') else 'unknown'
            })

        return voice_list

    def set_voice(self, voice_id):
        """
        Change the voice.

        Args:
            voice_id: Voice ID from get_available_voices()
        """
        self.engine.setProperty('voice', voice_id)

    def set_rate(self, rate):
        """
        Change speech rate.

        Args:
            rate: Words per minute (typically 100-300)
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """
        Change volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.engine.setProperty('volume', volume)


def test_audio_feedback():
    """Test audio feedback system."""
    print("Testing Audio Feedback System...")
    print("="*70)

    # Initialize
    audio = AudioFeedback(rate=180, volume=0.9)

    # List available voices
    print("\nAvailable voices:")
    voices = audio.get_available_voices()
    for i, voice in enumerate(voices[:5]):  # Show first 5
        print(f"  {i}: {voice['name']}")

    # Test basic speech
    print("\n[Test 1] Basic speech")
    audio.speak("BlindSpot AI audio system is working")
    time.sleep(2)

    # Test navigation
    print("\n[Test 2] Navigation announcement")
    audio.announce_safe_direction('left')
    time.sleep(2)

    # Test critical alert
    print("\n[Test 3] Critical alert")
    from inference.spatial_analyzer import SpatialObject
    critical_obj = SpatialObject(
        class_name='car',
        class_id=2,
        confidence=0.95,
        bbox=(100, 100, 200, 200),
        depth=0.9,
        distance=0.8,
        position='center',
        priority=1
    )
    audio.announce_critical_alert(critical_obj)
    time.sleep(3)

    # Test environment summary
    print("\n[Test 4] Environment summary")
    test_objects = [
        SpatialObject('person', 0, 0.9, (0, 0, 100, 100), 0.8, 2.0, 'left', 2),
        SpatialObject('chair', 56, 0.8, (0, 0, 100, 100), 0.6, 3.5, 'right', 3),
    ]
    audio.announce_environment_summary(test_objects)
    time.sleep(4)

    print("\n✓ Audio feedback test complete!")
    print("="*70)


if __name__ == "__main__":
    test_audio_feedback()
