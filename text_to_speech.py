"""
Text-to-Speech Module

This module handles converting text to speech using pyttsx3.
pyttsx3 is a FREE, offline text-to-speech library that works on Windows.

BEGINNER NOTE: This is the "voice" of the system - it reads text out loud!
"""

import pyttsx3
from config import SPEECH_RATE, SPEECH_VOLUME


class TextToSpeech:
    """
    A class to handle text-to-speech functionality.
    
    This uses pyttsx3, which is completely offline and free.
    No internet connection or API keys needed!
    """
    
    def __init__(self):
        """Initialize the text-to-speech engine."""
        print("Initializing text-to-speech engine...")
        
        try:
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', SPEECH_RATE)
            self.engine.setProperty('volume', SPEECH_VOLUME)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Try to set a good voice (prefer female voice if available)
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            
            print("✓ Text-to-speech engine initialized")
            
        except Exception as e:
            print(f"ERROR initializing TTS: {e}")
            self.engine = None
    
    def speak(self, text):
        """
        Speak the given text.
        
        Args:
            text: The text to speak
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            print("ERROR: TTS engine not initialized")
            return False
        
        if not text or text.strip() == "":
            print("⚠ No text to speak")
            return False
        
        try:
            print(f"Speaking: '{text}'")
            self.engine.say(text)
            self.engine.runAndWait()
            return True
            
        except Exception as e:
            print(f"ERROR speaking: {e}")
            return False
    
    def speak_async(self, text):
        """
        Speak text asynchronously (non-blocking).
        
        Args:
            text: The text to speak
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            print("ERROR: TTS engine not initialized")
            return False
        
        if not text or text.strip() == "":
            return False
        
        try:
            print(f"Speaking (async): '{text}'")
            self.engine.say(text)
            # Don't wait - let it speak in background
            return True
            
        except Exception as e:
            print(f"ERROR speaking: {e}")
            return False
    
    def stop(self):
        """Stop any ongoing speech."""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
    
    def set_rate(self, rate):
        """
        Set the speech rate.
        
        Args:
            rate: Words per minute (typically 100-200)
        """
        if self.engine:
            self.engine.setProperty('rate', rate)
            print(f"✓ Speech rate set to {rate} WPM")
    
    def set_volume(self, volume):
        """
        Set the speech volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine:
            volume = max(0.0, min(1.0, volume))  # Clamp to 0-1
            self.engine.setProperty('volume', volume)
            print(f"✓ Speech volume set to {volume}")
    
    def get_voices(self):
        """
        Get list of available voices.
        
        Returns:
            List of voice objects
        """
        if self.engine:
            return self.engine.getProperty('voices')
        return []
    
    def set_voice(self, voice_index):
        """
        Set the voice by index.
        
        Args:
            voice_index: Index of the voice to use
        """
        if self.engine:
            voices = self.get_voices()
            if 0 <= voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
                print(f"✓ Voice set to: {voices[voice_index].name}")
            else:
                print(f"ERROR: Invalid voice index {voice_index}")
    
    def close(self):
        """Clean up resources."""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        print("✓ TTS engine closed")


def test_tts():
    """
    Test function for text-to-speech.
    
    This will speak some test phrases to verify TTS is working.
    """
    print("\n" + "="*60)
    print("TEXT-TO-SPEECH TEST")
    print("="*60)
    
    tts = TextToSpeech()
    
    if not tts.engine:
        print("ERROR: Could not initialize TTS")
        return
    
    # List available voices
    print("\nAvailable voices:")
    voices = tts.get_voices()
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name}")
    
    # Test 1: Simple phrase
    print("\nTest 1: Speaking a simple phrase")
    tts.speak("Hello! I am your sign language assistant.")
    
    # Test 2: Alphabet
    print("\nTest 2: Speaking the alphabet")
    tts.speak("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z")
    
    # Test 3: Different rates
    print("\nTest 3: Different speech rates")
    
    print("Slow (100 WPM):")
    tts.set_rate(100)
    tts.speak("This is slow speech.")
    
    print("Normal (150 WPM):")
    tts.set_rate(150)
    tts.speak("This is normal speech.")
    
    print("Fast (200 WPM):")
    tts.set_rate(200)
    tts.speak("This is fast speech.")
    
    # Reset to default
    tts.set_rate(SPEECH_RATE)
    
    # Test 4: Sentence
    print("\nTest 4: Speaking a full sentence")
    tts.speak("The quick brown fox jumps over the lazy dog.")
    
    tts.close()
    
    print("\n" + "="*60)
    print("✓ TTS test completed")
    print("="*60)


if __name__ == "__main__":
    test_tts()
