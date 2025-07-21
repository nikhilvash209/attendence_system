import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime 

#Files
KNOWN_FACES_DIR = 'known_faces'
STUDENT_INFO_FILE = 'students_info.csv'
ATTENDANCE_FILE = 'attendance.csv'

# Student Informmation 
def load_student_info(student_file):
    df = pd.read_csv(student_file)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    df = df.loc[:, df.columns.str.contains('ID|Name|Department', case=False)]
    print("Detected Columns:", df.columns.tolist())
    return df


def load_known_faces(directory):
    known_encodings = []
    known_names = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts BGR TO RGB for face recognition
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    return known_encodings, known_names

def get_student_details(name, student_df):
    name_cleaned = name.strip().lower()
    for _, row in student_df.iterrows():
        student_name = row.get('Name')
        if isinstance(student_name, str) and student_name.strip().lower() == name_cleaned:
            student_id = row.get('ID', 'Unknown')
            department = row.get('Department', 'Unknown')
            return student_id, department
    return "Unknown", "Unknown"

# Marking attendence 
def mark_attendance(ID, Name, Department):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=['ID', 'Name', 'Department', 'Date', 'Time'])
        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    already_marked = ((df['ID'] == ID) & (df['Date'] == date_str)).any()


    # Prevent duplicate for this ID today
    if not already_marked:
        new_row = pd.DataFrame([{
            'ID': ID,
            'Name': Name,
            'Department': Department,
            'Date': date_str,
            'Time': time_str
        }])
        #pd.concat() is used to combine two or more dataframes or series together
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"Marked attendance for {Name} ({ID}) - {Department} at {date_str} {time_str}")
        return True, False
    else:
        print(f"Attendance already marked for {Name} ({ID}) - {Department} at {date_str} {time_str}")
        return False, True


#Main programe 

def main():
    print("Loading student info...")
    student_df = load_student_info(STUDENT_INFO_FILE)
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    print(f"Loaded faces for: {', '.join(known_names)}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open webcam.")
        return

    print("Camera live. Face the camera to mark attendance.")

    attendance_message = ""
    message_timestamp = None
    marked_or_skipped = False
    detection_start_time = datetime.now()
    recognizing = True
    face_detected_time = None
    display_labels = False
    label_name = ""
    label_id = ""
    label_branch = ""


    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read from webcam.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        #face_encoding is a numerical representation of a human face
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if face_encodings:
            if face_detected_time is None:
                face_detected_time = datetime.now()

                # Elapsed is a measurement of time — it tells you how much time has passed between two moments.
            elapsed = (datetime.now() - face_detected_time).total_seconds()

             # Only mark attendance after 1.5 sec of stable detection
            if elapsed > 1.5 and not marked_or_skipped:
                #face_encoding is a numerical representation of a human face
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                    name = "Unknown"
                    student_id = "Unknown"
                    branch = "Unknown"

                    if face_distances.any():
                        best_match = np.argmin(face_distances)
                        if matches[best_match]:
                            name = known_names[best_match]
                            student_id, branch = get_student_details(name, student_df)

                    if name != "Unknown":
                        marked, already = mark_attendance(student_id, name, branch)
                        attendance_message = "Attendance marked" if marked else "Attendance already marked"
                        message_timestamp = datetime.now()
                        marked_or_skipped = True
                        display_labels = True
                        label_name = name
                        label_id = student_id
                        label_branch = branch
                        break
        else:
            face_detected_time = None

        for face_location in face_locations:
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if display_labels:
                cv2.putText(frame, f"{label_name} ({label_id})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Department: {label_branch}", (left, bottom + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Unkown Face Detected", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                

        # Display messages

        if not marked_or_skipped:
            cv2.putText(frame, "Recognizing face...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif message_timestamp and (datetime.now() - message_timestamp).total_seconds() < 5:
            cv2.putText(frame, attendance_message, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        elif marked_or_skipped:
            break

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Exit after 10 seconds if nothing is detected

        if not marked_or_skipped and (datetime.now() - detection_start_time).total_seconds() > 10:
            print("No face detected. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
