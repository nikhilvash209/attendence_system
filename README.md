# 🎯 Face Recognition Attendance System

A Python-based face recognition attendance system using `face_recognition`, `OpenCV`, and `pandas`.

## 📌 Features
- Detects and recognizes known faces in real time using a webcam.
- Automatically logs attendance with timestamp and student details.
- Prevents duplicate entries per day.
- CSV-based student database and attendance logs.
- Shows clear labels and status messages on camera feed.

## 🖥️ Technologies Used
- Python 3.x
- OpenCV
- face_recognition
- NumPy
- Pandas

## 📁 Folder Structure
├── attendance_system_Version2.py # Main Python script

├── known_faces/ # Folder with face images (not uploaded)

├── students_info.csv # CSV with ID, Name, Department

├── attendance.csv # Output attendance log

├── .gitignore # Hides images or sensitive files



> Note: The `known_faces/` folder is excluded via `.gitignore` to protect privacy.

## 📋 How to Run
1. Install dependencies:
   ```bash
   pip install face_recognition opencv-python pandas numpy

2.Prepare:
Place student face images (named like John_Doe.jpg) inside known_faces/

Add their info in students_info.csv:
ID,Name,Department
101,John Doe,CSE
102,Jane Smith,ECE
