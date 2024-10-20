import cv2
import numpy as np
import time

def calculate_measures_4roi(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    corners4roi = [
        (0, height),
        (width / 3.2, height / 1.8),
        (width / 1.5, height / 1.5),
        (width, height)]
    
    return np.array([corners4roi], dtype=np.int32)

def region_of_interest(frame, corners4roi):
    mask = np.zeros_like(frame)
    match_mask_color = 255
    cv2.fillPoly(mask, corners4roi, match_mask_color) 
    masked_frame = cv2.bitwise_and(frame, mask)
    
    return masked_frame

def draw_the_lines(frame, lines):
    frame = np.copy(frame)
    blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_frame, (x1,y1), (x2,y2), (0, 255, 0), thickness=5)
    
    frame = cv2.addWeighted(frame, 0.9, blank_frame, 0.5, 0.0)
    
    return frame

def draw_filled_lane(frame, lines):
    if lines is None:
        return frame

    left_fit = []
    right_fit = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # Sağ şeridi tespit etmek için koşullar
            if slope < -0.5 and slope > -2.0 and x1 < frame.shape[1] / 2 and x2 < frame.shape[1] / 2:
                left_fit.append((x1, y1))
                left_fit.append((x2, y2))
            elif slope > 0.5 and slope < 2.0 and x1 > frame.shape[1] / 2 and x2 > frame.shape[1] / 2:
                right_fit.append((x1, y1))
                right_fit.append((x2, y2))

    if len(left_fit) > 0 and len(right_fit) > 0:
        left_fit = np.array(left_fit)
        right_fit = np.array(right_fit)
        
        left_fit = left_fit[left_fit[:, 1].argsort()]
        right_fit = right_fit[right_fit[:, 1].argsort()]
        
        left_line_pts = np.vstack((left_fit[0], left_fit[-1]))
        right_line_pts = np.vstack((right_fit[0], right_fit[-1]))
        
        pts = np.vstack((left_line_pts, right_line_pts[::-1]))
        pts = np.array([pts], dtype=np.int32)
        
        cv2.fillPoly(frame, pts, (0, 255, 0))

    return frame

def detect_lines_and_process(frame):
    blurred_frame = cv2.GaussianBlur(frame, (9, 9), 0.8)
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2GRAY)
    canny_frame = cv2.Canny(gray_frame, 50, 150)
    cropped_frame = region_of_interest(canny_frame, calculate_measures_4roi(frame))
    lines = cv2.HoughLinesP(cropped_frame,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=10,
                            maxLineGap=250)
    frame_with_lines = draw_the_lines(frame, lines)
    frame_with_filled_lane = draw_filled_lane(frame_with_lines, lines)
    
    return frame_with_filled_lane

cap = cv2.VideoCapture("/Users/izzetozguronder/Desktop/serit_takip_sistemi-main/Yol VideolarÄ± Mersin_Silifke  #yolvideolarÄ± #silifke #cubamusic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Original Road", frame)
    
    frame = detect_lines_and_process(frame)
    time.sleep(0.01)
    
    cv2.imshow("Processed Road", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
