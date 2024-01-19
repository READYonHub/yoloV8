import cv2
import numpy as np

# Kamera inicializálása
cap = cv2.VideoCapture(0)

# YOLO modell betöltése
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg") # ket fájl betöltése
classes = []
with open("./coco.names", "r") as f: # megnyitja a names fájlt
    classes = [line.strip() for line in f.readlines()] # szétszedi sorokra és belerakja a tömbbe
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # a neron hálózatnak a rétegeit vizsgálja, hogy mi van benne
font = cv2.FONT_HERSHEY_PLAIN # betutipus a kiirasokhoz

# frame feldolgozása
while True:
    # Kamera képkockájának beolvasása
    ret, frame = cap.read()

    # Kamerakep kepkockainak méreteinek lekérdezése
    height, width, channels = frame.shape

    # YOLO-alapú objektumdetekció
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # ezt atalakitja bloba
    net.setInput(blob) # es ezt a blobot fogja beallitani a neuronhálózatnak bemenetnek
    outs = net.forward(output_layers) # vegig megy a neuronhálón a blobbal

    # Képernyőn információ megjelenítése / Az algoritmus bizalmi értékének lekérése
    for out in outs: 
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Ha az algoritmus biztos abban, hogy egy objektumot észlelt
            if confidence > 0.3: # a 0,3-nál nagyobb beazonosítható objektumokat vegyük ki
                # Objektum pozíciójának kiszámolása
                center_x = int(detection[0] * width) # x koordinata
                center_y = int(detection[1] * height) # y koordinata
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Téglalap koordináták kiszámolása
                x = int(center_x - w/2) # eltoljuk a kozepétől félel különben rossz helyre rajzolja a téglalapot
                y = int(center_y - h/2)

                # Téglalap kirajzolása a képre
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
                
                # Az objektum nevének és bizalmi értékének kiírása a képre
                cv2.putText(frame, classes[class_id], (x, y), font, 1, (255, 255, 255), 1)
                
                break # ha megtalalta a beazonosított objektumot akkor lépjen ki, ezzel tudjuk elkerülni, hogy ne rajzoljon körbe egy és ugyanazat az felismert objektumot többször

    # A képkocka megjelenítése
    cv2.imshow('frame', frame)
    
    # Kilépés a 'q' gomb megnyomásával
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

# Kamera felszabadítása
cap.release()

# Az összes ablak bezárása
cv2.destroyAllWindows()
