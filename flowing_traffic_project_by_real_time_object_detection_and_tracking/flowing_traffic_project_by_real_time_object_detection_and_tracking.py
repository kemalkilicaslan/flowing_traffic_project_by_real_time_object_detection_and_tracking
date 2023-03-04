# Flowing Traffic Project By Real Time Object Detection And Tracking
# Gerçek Zamanlı Nesne Tespiti ve Takibi ile Akan Trafik Projesi

import cv2

# read video file
# video dosyasını okuyun
cap = cv2.VideoCapture('traffic.mp4')

# install HOG recognizers for pedestrian detection
# yaya algılama için HOG tanıyıcılarını yükleyin
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # get a frame from video
    # video'dan bir kare alın
    ret, frame = cap.read()

    # exit the loop if the square is empty
    # kare boş ise döngüden çıkın
    if not ret:
        break

    # convert original frame to grayscale for pedestrian detection
    # yaya algılama için orijinal kareyi gri tonlamaya dönüştürün
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect pedestrians using HOG recognizers
    # HOG tanıyıcılarını kullanarak yaya algılama yapın
    pedestrians, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draw detected pedestrian zones
    # algılanan yaya bölgeleri çizdirin
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the result frame
    # sonuç karesini gösterin
    cv2.imshow('Pedestrian Detection', frame)

    # exit when 'q' key is pressed
    # 'q' tuşuna basıldığında çıkın
    if cv2.waitKey(1) == ord('q'):
        break

# let's end the process.
# işlemi sonlandıralım.
cap.release()
cv2.destroyAllWindows()


# This code detects pedestrians in real time by taking frames from a video file named "traffic.mp4" and displays the results on the screen.
# Bu kod, "traffic.mp4" adlı bir video dosyasından kareler alarak gerçek zamanlı olarak yaya tespiti yapar ve sonuçları ekranda gösterir.