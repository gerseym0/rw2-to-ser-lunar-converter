Markdown
# RW2 to SER Converter

A Python script designed to convert Panasonic Lumix raw image sequences (`.RW2`) into a single `.SER` video file format. This tool is specifically built for astrophotographers who shoot lunar or planetary sequences in burst mode and need to process them in stacking software like AutoStakkert!3 (AS!3).

## Features

*   **Native Raw Bayer Output:** Packages raw sensor data into a 16-bit SER container without performing internal debayering, preserving the original data for processing in specialized astrophotography software.
*   **Automatic Lunar Disk Detection:** Analyzes the first few frames to find the moon's position using image moments.
*   **Intelligent Cropping:** Automatically crops the frames around the detected lunar disk with a user-defined padding. This significantly reduces the final SER file size and drastically speeds up the alignment and stacking process in AS!3.
*   **Bayer Pattern Alignment:** Ensures that crop dimensions are always even numbers, preventing critical Bayer matrix shifts that cause color artifacting during debayering.

## Requirements

The script requires Python 3 and the following libraries:

```bash
pip install rawpy numpy scipy
Usage
Place all your .RW2 sequence files into a single directory.

Open rw2_to_ser.py in a text editor and update the configuration section at the top of the script:

INPUT_FOLDER: The path to the folder containing your RW2 files.

OUTPUT_FILE: The path and filename for the resulting SER file.

ENABLE_CROP: Set to True to enable automatic moon detection and cropping (recommended).

Run the script:

Bash
python rw2_to_ser.py
Alternatively, you can pass the input folder and output file directly via command-line arguments:

Bash
python rw2_to_ser.py /path/to/rw2/folder /path/to/output.ser
Workflow in AutoStakkert!3
Because this script exports raw Bayer data, you must configure AutoStakkert!3 correctly to interpret the colors and avoid a "grid" pattern on the final stack:

Open the generated .ser file in AS!3.

In the top menu, go to Color -> Force Bayer RGGB (This is the standard pattern for Panasonic S5; adjust if using a different camera).

In the right-hand panel, under Advanced Settings, select Drizzle 1.5x (or 3.0x).

Note: Using Drizzle in AS!3 on raw Bayer data automatically triggers the "Bayer Drizzle" algorithm, which is necessary to correctly reconstruct the color image from the sensor data without introducing debayering artifacts.

License
MIT License. Free to use, modify, and distribute.
