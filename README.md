# Anleitung

## Doku: Unimodel

***dataset_bioreaktorLuft.py***:  Erstellen Sie den Datensatz (Volumenstrom als Anomalie) und speichern Sie den Path zu einer csv-Datei. Das Format ist (Path, label für GeoTrans)  
    *Arguments:* 1.root(): Path für Datensatz 2. size: Die Dimension des Bildes 3. mode: Train, Test: Validierung von Modellen mit anomalen Daten ,Vali: Validierung von Modellen mit nomalen Daten

***dataset_bioreaktorSpeed.py***:  Erstellen Sie den Datensatz (Rührdrehzahl als Anomalie) und speichern Sie den Path zu einer csv-Datei. Das Format ist (Path, label für GeoTrans)  
    *Arguments:* 1.root(): Path für Datensatz 2. size: Die Dimension des Bildes 3. mode: Train, testbig: Validierung von Modellen mit anomalen Daten(Drehzahl>400) ,Vali: Validierung von Modellen mit nomalen Daten, testsmall: Validierung von Modellen mit anomalen Daten(Drehzahl<400)

***Model.py***: Erstellen Sie den WideResNet  
    **Arguments*:*  1. N: Tiefe des Netzes. Für dataset_bioreajtorLuft.py N=10, Für dataset_bioreajtorSpeed.py N=16. 2. k: Breite des Netzes  . Für dataset_bioreajtorLuft.py k=8, Für dataset_bioreajtorSpeed.py N=8  3. num_classes Anzahl der Ausgangsknoten

***metric.py***:  Erhalten Sie Roc-Kurve, Konfusionsmatrix und AUC-Werte
        **load_data()**: Berechnen Sie die Anomaliescore.
            *Arguments:* 1. dataname: Wählen Sie aus, zu welcher Ausnahme die berechneten Metriken gehören. 'Speed300': Drehzahl<400 als Anomalie, 'Speed500':Drehzahl>400 als Anomalie. 'Luft': Volumenstrom als Anomalie.
            *Return:* 1, TPFNTNFP_label: Ziel für Anomaliescore0: Anomalie 1: normale Daten, 2, TPFNTNFP_prob: Anomaliescore
        **draw_roc()**: Zeichnen von Roc-Kurven
            *Arguments:* 1. Ziel für Anomaliescore label: 0: Anomalie 1: normale Daten  2. prob: Anomaliescore. 3. name: name für Bild
        **cf_matrix:** Erhalten Sie die Konfusionsmatrix

***pre_process.py***:  Vorverarbeitung der Bilder
    **cut_picture()**: Ein Bild auswählen und den ausgeschnittenen Bereich mit der Maus markieren  
    **pre_process()**: Schneiden Sie das Bild entsprechend dem oben gespeicherten Bereich aus. Und speichern Sie in dem entsprechenden Ordner

***drawheatmap.py***: Erhaltrn Sie die Heatmap(CAM)
    *Arguments:* 1. model: Lesen von Modellen und Gewichten. 2, img_path: Die Heatmap dieses Bildes erhalten. 3. save_path

***trainuniluft, trainunispeed***: Das Modell wird trainiert. Das Modell wird einmal pro Trainingsepoche getestet, um die Klassifizierungsgenauigkeit für normale und abnormale Daten zu erhalten. Visualisierung über visdom

## Doku: 
Multimodel: In dieser Dokumentation wird ein multimodales Modell mit geometrischer Transformation trainiert und ausgewertet.

***dataset_bioreaktorMulti***: Erstellen Sie den Datensatz (die nicht mit dem Bild übereinstimmten Prozessparameter als Anomalie) und speichern Sie den Path, Prozesparameter, label, zu einer csv-Datei. Das Format ist (Path, Prozessparameter, label für GeoTrans).   
   *Arguments:* 1.root(): Path für Datensatz 2. size: Die Dimension des Bildes 3. mode: Train, Test: Validierung von Modellen mit anomalen Daten(Bilder mit falsche Prozessparameter) ,Vali: Validierung von Modellen mit nomalen Daten (Bilder mit richtige Prozessparameter)

***Multi_Model.py***: Erstellen Sie den WideResNet  
    **Arguments*:*  1. N: Tiefe des CNN Netzes. (N=10). 2. k: Breite des CNN Netzes. k=8  3. num_classes Anzahl der Ausgangsknoten 4. res_factor: Breite des MLP =10 5. dropoutrate Dropout für MLP Nezte = 0  

***metric.py***:  Erhalten Sie Roc-Kurve, Konfusionsmatrix und AUC-Werte
        **load_data()**: Berechnen Sie die Anomaliescore.
            *Arguments:* 1. dataname: Wählen Sie aus, zu welcher Ausnahme die berechneten Metriken gehören. 'ModelZustand': Parameter, die nicht mit dem Bild übereinstimmen, als Ausnahmen verarbeiten.
            *Return:* 1, TPFNTNFP_label: Ziel für Anomaliescore0: Anomalie 1: normale Daten, 2, TPFNTNFP_prob: Anomaliescore
        **draw_roc()**: Zeichnen von Roc-Kurven
            *Arguments:* 1. Ziel für Anomaliescore label: 0: Anomalie 1: normale Daten  2. prob: Anomaliescore. 3. name: name für Bild
        **cf_matrix:** Erhalten Sie die Konfusionsmatrix

***train_multi***: Das Modell wird trainiert. Das Modell wird einmal pro Trainingsepoche getestet, um die Klassifizierungsgenauigkeit für normale und abnormale Daten zu erhalten. Visualisierung über visdom

## Doku: MLflow
In dieser Dokumentation wird ein multimodales Modell mit geometrischer Transformation trainiert und ausgewertet. Hier werden die Hyperparameter durch mlflow optimiert

***Multi_Model_NoGeo.py***: Erstellen Sie den WideResNet  
    **Arguments*:*  1. depth N : Tiefe des CNN Netzes,2. num_classes: Anzahl der Ausgangsknoten, 3. widen_factor; Breite des CNN Netzes 4. res_factor: Breite des MLP Netzes, 5. dropRate: Dropout für CNN, 6. dropRateDense: Drop für MLP, 7. Fcn_depth: Tiefe für MLP Netzes
 
***get_dataloader.py***: Holen Sie sich die Trainingsdatenbank und die Testdatenbank 
      **Arguments**: csv_train='Train50', csv_test='Vali50': Differz zwischen Normalbetrieb und Anomalie sind (50,5). csv_train='Train100', csv_test='Vali100': Differz    zwischen Normalbetrieb und Anomalie sind (100,10). csv_train='Train200', csv_test='Vali200': Differz zwischen Normalbetrieb und Anomalie sind (200,20).     csv_train='Train300', csv_test='Vali300': Differz zwischen Normalbetrieb und Anomalie sind (300,30).
  
**dataset_bioreaktorMulti_NoGeo.py:**Erstellen Sie den Datensatz (die nicht mit dem Bild übereinstimmten Prozessparameter als Anomalie) und speichern Sie den Path, Prozesparameter, label, zu einer csv-Datei. Das Format ist (Path, Prozessparameter, label(normal=0, ANnoamlie=1)).

**train_test.py**: Trainingsverlauf und Testverlauf für das Modell. Das Modell wird einmal pro Trainingsepoche getestet, um die Klassifizierungsgenauigkeit für normale und abnormale Daten zu erhalten. Visualisierung über visdom.
   **test_model()**: 
      *Arguments:*1. model: Modell mit Gewicht, 2.device, 3.test_loader
      *return*: 1. anormalacc: Klassifizierungsgenauigkeit für Anomalie, 2.torch.ones_like(total_label) - total_label: Anomaliescore, 3.torch.ones_like(total_pred) - total_pred: Labels für Anomaliescore, 4. avg_val_loss: Durchschnittlicher Verlust
    **train_model()**:
      *Arguments*:1.model, 2.device, 3.train_loader, 4.optimizer, 5.epoch, 6.epochs, 7.global_step: Gesamtzahl der Iterationzahl(letzte Epoche)
      *return*: global_step: Gesamtzahl der Iterationzahl(diese Epoche), avg_train_loss
  
**HP_Optim.py**: Hyperparametrische Optimierung durchführen, Verlust, bestes Modell, Metriken(ROC,f1,Konfusionsmatrix) für besten Modell und AUC für jedes epcoh speichern 
    
