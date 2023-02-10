# Anleitung

## Doku: Unimodel

dataset_bioreaktorLuft.py:  Erstellen Sie den Datensatz (Volumenstrom als Anomalie) und speichern Sie den Path zu einer csv-Datei. Das Format ist (Path, label für GeoTrans)  
    Arguments: 1.root(): Path für Datensatz 2. size: Die Dimension des Bildes 3. mode: Train, Test: Validierung von Modellen mit anomalen Daten ,Vali: Validierung von Modellen mit nomalen Daten

dataset_bioreaktorSpeed.py:  Erstellen Sie den Datensatz (Rührdrehzahl als Anomalie) und speichern Sie den Path zu einer csv-Datei. Das Format ist (Path, label für GeoTrans)  
    Arguments: 1.root(): Path für Datensatz 2. size: Die Dimension des Bildes 3. mode: Train, testbig: Validierung von Modellen mit anomalen Daten(Drehzahl>400) ,Vali: Validierung von Modellen mit nomalen Daten, testsmall: Validierung von Modellen mit anomalen Daten(Drehzahl<400)

Model.py: Erstellen Sie den WideResNet  
    Arguments:  1. N: Tiefe des Netzes. Für dataset_bioreajtorLuft.py N=10, Für dataset_bioreajtorSpeed.py N=16. 2. k: Breite des Netzes  . Für dataset_bioreajtorLuft.py k=8, Für dataset_bioreajtorSpeed.py N=8  3. num_classes Anzahl der Ausgangsknoten

metric.py:  Erhalten Sie Roc-Kurve, Konfusionsmatrix und AUC-Werte
        load_data(): Berechnen Sie die Anomaliescore.
            Arguments: 1. dataname: Wählen Sie aus, zu welcher Ausnahme die berechneten Metriken gehören. 'Speed300': Drehzahl<400 als Anomalie, 'Speed500':Drehzahl>400 als Anomalie. 'Luft': Volumenstrom als Anomalie.
            Return: 1, TPFNTNFP_label: Ziel für Anomaliescore0: Anomalie 1: normale Daten, 2, TPFNTNFP_prob: Anomaliescore
        draw_roc(): Zeichnen von Roc-Kurven
            Arguments: 1. Ziel für Anomaliescore label: 0: Anomalie 1: normale Daten  2. prob: Anomaliescore. 3. name: name für Bild
        cf_matrix: Erhalten Sie die Konfusionsmatrix

pre_process.py:  Vorverarbeitung der Bilder
    cut_picture(): Ein Bild auswählen und den ausgeschnittenen Bereich mit der Maus markieren  
    pre_process(): Schneiden Sie das Bild entsprechend dem oben gespeicherten Bereich aus. Und speichern Sie in dem entsprechenden Ordner

drawheatmap.py: Erhaltrn Sie die Heatmap(CAM)
    Arguments: 1. model: Lesen von Modellen und Gewichten. 2, img_path: Die Heatmap dieses Bildes erhalten. 3. save_path

trainuniluft, trainunispeed: Das Modell wird trainiert. Das Modell wird einmal pro Trainingsepoche getestet, um die Klassifizierungsgenauigkeit für normale und abnormale Daten zu erhalten. Visualisierung über visdom

## Doku: Multimodel

## Doku: MLflow
